class FakePredictor:
    available=True
    model_version='fake'
    def predict(self, source_ar, candidate_en):
        if 'BAD' in candidate_en:
            return {'available':True,'confidence':0.9,'severity':'S3','findings':[{'label':'NEGATION_FLIP','severity':'S3','confidence':0.9,'origin':'model'}],'model_version':'fake'}
        return {'available':True,'confidence':0.95,'severity':'S0','findings':[],'model_version':'fake'}


def test_evaluate_rows_computes_measured_metrics_from_predictor():
    from training.evaluate import evaluate_rows
    rows=[{'source_ar':'a','candidate_en':'GOOD','labels':['FAITHFUL'],'severity':'S0'},{'source_ar':'a','candidate_en':'BAD','labels':['NEGATION_FLIP'],'severity':'S3'}]
    result=evaluate_rows(rows,FakePredictor())
    assert result['status']=='ok'; assert result['rows']==2; assert result['critical_drift_recall']==1.0; assert result['false_safe_rate']==0.0; assert result['macro_f1'] > 0


def test_evaluate_rows_false_safe_rate_uses_faithful_label_by_name_not_column_position():
    from training.evaluate import evaluate_rows
    class FaithfulOnlyPredictor:
        available=True
        model_version='fake'
        def predict(self, source_ar, candidate_en):
            return {'available':True,'confidence':0.99,'severity':'S0','findings':[],'model_version':'fake'}
    rows=[{'source_ar':'a','candidate_en':'bad','labels':['NEGATION_FLIP'],'severity':'S3'}]
    result=evaluate_rows(rows,FaithfulOnlyPredictor(),label_names=['NEGATION_FLIP','FAITHFUL'])
    assert result['false_safe_rate'] == 1.0
    assert result['critical_drift_recall'] == 0.0
