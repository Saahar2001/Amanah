from fastapi.testclient import TestClient
from amanah_engine.schemas import AnalyzeResponse

class FakeService:
    def __init__(self,mode='pass'): self.mode=mode
    def analyze(self,request):
        if self.mode=='critical':
            from amanah_engine.schemas import DriftFinding
            return AnalyzeResponse(decision='CRITICAL',integrity_score=55,severity='S3',confidence=.99,drifts=[DriftFinding(label='NEGATION_FLIP',severity='S3',confidence=.99,origin='rule')],needs_human_review=True,model_version='test',reference_status='verified')
        if self.mode=='abstain': return AnalyzeResponse(decision='ABSTAIN',integrity_score=0,severity='S0',confidence=0,drifts=[],needs_human_review=True,model_version='unavailable',reference_status='missing',notes=['missing'])
        return AnalyzeResponse(decision='PASS',integrity_score=100,severity='S0',confidence=.95,drifts=[],needs_human_review=False,model_version='test',reference_status='verified')

def client(mode='pass'):
    from api.main import create_app
    return TestClient(create_app(service=FakeService(mode)))

def payload(): return {'source_type':'quran','source_ar':'لا إكراه في الدين','candidate_en':'There is no compulsion in religion.','ayah_id':'2:256'}

def test_health_version_and_valid_analysis():
    c=client(); assert c.get('/health').json()['status']=='ok'; assert 'version' in c.get('/version').json(); r=c.post('/v1/analyze',json=payload()); assert r.status_code==200; assert r.json()['decision']=='PASS'

def test_malformed_request_gets_422(): assert client().post('/v1/analyze',json={'source_type':'hadith','source_ar':'x','candidate_en':'y'}).status_code==422

def test_abstain_and_finding_contracts_are_preserved():
    a=client('abstain').post('/v1/analyze',json=payload()).json(); assert a['decision']=='ABSTAIN' and a['needs_human_review'] is True
    critical=client('critical').post('/v1/analyze',json=payload()).json(); assert critical['decision']=='CRITICAL'; assert critical['drifts'][0]['label']=='NEGATION_FLIP'

def test_optional_bearer_token_protects_analysis_endpoint():
    from api.main import create_app
    protected=TestClient(create_app(service=FakeService(),api_token='secret-token')); assert protected.post('/v1/analyze',json=payload()).status_code==401; allowed=protected.post('/v1/analyze',json=payload(),headers={'Authorization':'Bearer secret-token'}); assert allowed.status_code==200 and allowed.json()['decision']=='PASS'

class NotReadyService(FakeService):
    def readiness(self): return {'ready':False,'model_available':False,'reference_count':0}

def test_ready_returns_503_when_model_or_references_are_not_loaded():
    from api.main import create_app
    r=TestClient(create_app(service=NotReadyService())).get('/ready'); assert r.status_code==503 and r.json()['detail']['ready'] is False

def test_ready_returns_component_status_when_service_is_ready():
    class ReadyService(FakeService):
        def readiness(self): return {'ready':True,'model_available':True,'reference_count':6236}
    from api.main import create_app
    r=TestClient(create_app(service=ReadyService())).get('/ready'); assert r.status_code==200 and r.json()=={'ready':True,'model_available':True,'reference_count':6236}
