from data.models import DriftLabel, Severity

def labels(findings): return {f.label for f in findings}

def test_high_precision_rules_detect_core_drift_types():
    from amanah_engine.semantic_rules import run_rules
    assert DriftLabel.NEGATION_FLIP in labels(run_rules('They do not transgress.','They do transgress.'))
    assert DriftLabel.MODALITY_SHIFT in labels(run_rules('They may give charity.','They must give charity.'))
    assert DriftLabel.QUANTIFIER_CHANGE in labels(run_rules('Some people know.','All people know.'))
    assert DriftLabel.CONDITION_LOSS in labels(run_rules('If they repent, then forgive them.','Forgive them.'))

def test_coverage_anomaly_is_review_not_automatic_critical():
    from amanah_engine.semantic_rules import run_rules
    omission=[f for f in run_rules('This is a sufficiently long reference sentence with several semantic units.','Short.') if f.label==DriftLabel.OMISSION]
    assert omission and omission[0].severity==Severity.S2
