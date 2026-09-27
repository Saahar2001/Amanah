from amanah_engine.schemas import DriftFinding

def f(label,severity,confidence,origin='model'): return DriftFinding(label=label,severity=severity,confidence=confidence,origin=origin)

def test_critical_rule_overrides_model_and_clean_passes():
    from amanah_engine.scoring import fuse_decision
    critical=fuse_decision(model_findings=[],rule_findings=[f('NEGATION_FLIP','S3',1.0,'rule')],model_available=True,model_confidence=0.9); assert critical.decision.value=='CRITICAL' and critical.needs_human_review is True
    clean=fuse_decision(model_findings=[],rule_findings=[],model_available=True,model_confidence=0.95); assert clean.decision.value=='PASS' and clean.integrity_score>=90

def test_low_confidence_abstains_and_disagreement_reviews():
    from amanah_engine.scoring import fuse_decision
    assert fuse_decision(model_findings=[],rule_findings=[],model_available=True,model_confidence=0.2).decision.value=='ABSTAIN'
    assert fuse_decision(model_findings=[f('OMISSION','S2',0.8)],rule_findings=[],model_available=True,model_confidence=0.8).decision.value=='REVIEW'
