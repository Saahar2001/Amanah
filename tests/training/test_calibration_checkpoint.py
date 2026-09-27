from pathlib import Path


class FakeScorePredictor:
    model_version = 'fake'
    available = True
    def predict_scores(self, source_ar, candidate_en):
        if candidate_en == 'a': return {'available':True,'probabilities':[0.9,0.2]}
        if candidate_en == 'b': return {'available':True,'probabilities':[0.8,0.4]}
        if candidate_en == 'c': return {'available':True,'probabilities':[0.1,0.8]}
        return {'available':True,'probabilities':[0.2,0.7]}


def test_calibrate_predictor_saves_per_label_thresholds(tmp_path: Path):
    from training.calibrate_thresholds import calibrate_predictor
    rows=[
        {'source_ar':'x','candidate_en':'a','labels':['FAITHFUL']},
        {'source_ar':'x','candidate_en':'b','labels':['FAITHFUL']},
        {'source_ar':'x','candidate_en':'c','labels':['NEGATION_FLIP']},
        {'source_ar':'x','candidate_en':'d','labels':['NEGATION_FLIP']},
    ]
    path=tmp_path/'thresholds.json'
    result=calibrate_predictor(rows,FakeScorePredictor(),['FAITHFUL','NEGATION_FLIP'],path,candidates=[0.3,0.5,0.75])
    assert result == [0.75,0.5]
    assert path.exists()
