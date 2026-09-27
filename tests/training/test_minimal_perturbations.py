import json
from pathlib import Path


def test_minimal_perturbations_are_critical_and_lexically_close():
    from training.baselines import lexical_token_similarity
    rows=json.loads(Path('examples/minimal_perturbations.json').read_text())
    assert {r['label'] for r in rows} == {'NEGATION_FLIP','QUANTIFIER_CHANGE','MODALITY_SHIFT'}
    assert all(r['severity']=='S3' for r in rows)
    sims=[lexical_token_similarity(r['trusted'],r['candidate']) for r in rows]
    assert min(sims) >= 0.6
