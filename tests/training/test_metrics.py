import numpy as np


def test_metrics_include_critical_recall_false_safe_and_macro_f1():
    from training.metrics import compute_drift_metrics
    y_true=np.array([[1,0,0],[0,1,0],[0,0,1],[0,1,0]],dtype=int)
    y_pred=np.array([[1,0,0],[0,1,0],[0,0,0],[0,0,0]],dtype=int)
    severity_true=np.array([0,3,3,2]); severity_pred=np.array([0,3,0,0])
    metrics=compute_drift_metrics(y_true,y_pred,severity_true,severity_pred,critical_mask=(severity_true==3))
    assert metrics['critical_drift_recall'] == 0.5
    assert metrics['false_safe_rate'] == 0.5
    assert 0 <= metrics['macro_f1'] <= 1
    assert 0 <= metrics['severity_macro_f1'] <= 1
    assert len(metrics['per_label_f1']) == 3


def test_false_safe_rate_treats_faithful_prediction_as_safe_on_critical_sample():
    from training.metrics import compute_drift_metrics
    y_true=np.array([[0,1,0]],dtype=int); y_pred=np.array([[1,0,0]],dtype=int); severity_true=np.array([3]); severity_pred=np.array([0])
    metrics=compute_drift_metrics(y_true,y_pred,severity_true,severity_pred,critical_mask=np.array([True]),faithful_index=0)
    assert metrics['false_safe_rate'] == 1.0
    assert metrics['critical_drift_recall'] == 0.0
