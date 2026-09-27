from amanah_engine.schemas import AnalyzeRequest

class FakeRefs:
    def __init__(self,found=True): self.found=found
    def lookup(self,ayah_id):
        if not self.found: return None
        return {'ayah_id':ayah_id,'source_ar':'They do not transgress.','references_en':['They do not transgress.'],'provenance_verified':True}
class FakeClassifier:
    available=True; model_version='test-model'
    def predict(self,source_ar,candidate_en): return {'available':True,'confidence':0.9,'findings':[],'model_version':self.model_version}
class DownClassifier:
    available=False; model_version='unavailable'
    def predict(self,*args,**kwargs): return {'available':False,'confidence':0.0,'findings':[],'model_version':'unavailable','error':'not loaded'}
def request(candidate='They do transgress.'): return AnalyzeRequest(source_type='quran',source_ar='They do not transgress.',candidate_en=candidate,ayah_id='1:1')

def test_service_uses_reference_and_rules():
    from amanah_engine.service import AmanahAnalysisService
    res=AmanahAnalysisService(FakeClassifier(),FakeRefs()).analyze(request()); assert res.decision.value=='CRITICAL' and res.reference_status=='verified' and any(x.label.value=='NEGATION_FLIP' for x in res.drifts)

def test_service_missing_reference_or_model_unavailable_abstains():
    from amanah_engine.service import AmanahAnalysisService
    missing=AmanahAnalysisService(FakeClassifier(),FakeRefs(False)).analyze(request()); assert missing.decision.value=='ABSTAIN' and missing.needs_human_review
    down=AmanahAnalysisService(DownClassifier(),FakeRefs()).analyze(request('They do not transgress.')); assert down.decision.value=='ABSTAIN' and down.needs_human_review

class MultiRefs:
    def lookup(self,ayah_id): return {'ayah_id':ayah_id,'source_ar':'مصدر','references_en':['They may give charity.','They can give charity.'],'provenance_verified':True}

def test_service_reference_envelope_does_not_flag_accepted_reference_variation():
    from amanah_engine.service import AmanahAnalysisService
    req=AnalyzeRequest(source_type='quran',source_ar='مصدر',candidate_en='They can give charity.',ayah_id='1:1'); res=AmanahAnalysisService(FakeClassifier(),MultiRefs()).analyze(req); assert all(x.label.value!='MODALITY_SHIFT' for x in res.drifts) and res.decision.value=='PASS'

class CanonicalMismatchRefs:
    def lookup(self,ayah_id): return {'ayah_id':ayah_id,'source_ar':'النص العربي الموثوق','references_en':['Trusted translation.'],'provenance_verified':True}

def test_service_abstains_when_request_arabic_does_not_match_canonical_reference():
    from amanah_engine.service import AmanahAnalysisService
    res=AmanahAnalysisService(FakeClassifier(),CanonicalMismatchRefs()).analyze(AnalyzeRequest(source_type='quran',source_ar='نص عربي مختلف',candidate_en='Trusted translation.',ayah_id='1:1')); assert res.decision.value=='ABSTAIN' and res.reference_status=='source_mismatch' and res.needs_human_review is True

class UnverifiedProvenanceRefs:
    def lookup(self,ayah_id): return {'ayah_id':ayah_id,'source_ar':'مصدر','references_en':['Trusted translation.']}

def test_service_abstains_when_reference_provenance_is_not_verified():
    from amanah_engine.service import AmanahAnalysisService
    res=AmanahAnalysisService(FakeClassifier(),UnverifiedProvenanceRefs()).analyze(AnalyzeRequest(source_type='quran',source_ar='مصدر',candidate_en='Trusted translation.',ayah_id='1:1')); assert res.decision.value=='ABSTAIN' and res.reference_status=='unverified_provenance'

class AutoMatchRefs:
    def lookup(self,ayah_id): return None
    def match_source_ar(self,source_ar):
        if source_ar=='لا إكراه في الدين': return {'ayah_id':'2:256','source_ar':'لا إكراه في الدين','references_en':['There is no compulsion in religion.'],'provenance_verified':True}
        return None

def test_service_can_resolve_reference_from_canonical_arabic_when_ayah_id_missing():
    from amanah_engine.service import AmanahAnalysisService
    res=AmanahAnalysisService(FakeClassifier(),AutoMatchRefs()).analyze(AnalyzeRequest(source_type='quran',source_ar='لا إكراه في الدين',candidate_en='There is no compulsion in religion.')); assert res.reference_status=='verified' and res.decision.value=='PASS'

def test_service_abstains_when_neither_id_nor_source_can_be_resolved():
    from amanah_engine.service import AmanahAnalysisService
    res=AmanahAnalysisService(FakeClassifier(),AutoMatchRefs()).analyze(AnalyzeRequest(source_type='quran',source_ar='نص غير موجود',candidate_en='Unknown')); assert res.decision.value=='ABSTAIN' and res.reference_status=='missing'
