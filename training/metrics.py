from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def critical_drift_recall(y_true: np.ndarray, y_pred: np.ndarray, critical_mask: np.ndarray) -> float:
    idx = np.asarray(critical_mask, dtype=bool)
    if idx.sum() == 0:
        return 0.0
    true_pos_by_sample = ((y_true[idx] & y_pred[idx]).sum(axis=1) > 0)
    return float(true_pos_by_sample.mean())


def false_safe_rate(y_true: np.ndarray, y_pred: np.ndarray, critical_mask: np.ndarray, *, faithful_index: int | None = 0) -> float:
    idx = np.asarray(critical_mask, dtype=bool)
    if idx.sum() == 0:
        return 0.0
    drift_pred = np.asarray(y_pred[idx], dtype=int).copy()
    if faithful_index is not None and 0 <= faithful_index < drift_pred.shape[1]:
        drift_pred[:, faithful_index] = 0
    predicted_safe = (drift_pred.sum(axis=1) == 0)
    return float(predicted_safe.mean())


def compute_drift_metrics(y_true, y_pred, severity_true, severity_pred, critical_mask, *, faithful_index: int | None = 0) -> dict:
    y_true = np.asarray(y_true, dtype=int); y_pred = np.asarray(y_pred, dtype=int)
    severity_true=np.asarray(severity_true); severity_pred=np.asarray(severity_pred)
    per_label=f1_score(y_true,y_pred,average=None,zero_division=0).tolist()
    return {
        "macro_f1": float(f1_score(y_true,y_pred,average="macro",zero_division=0)),
        "per_label_f1": [float(x) for x in per_label],
        "severity_macro_f1": float(f1_score(severity_true,severity_pred,average="macro",zero_division=0)),
        "critical_drift_recall": critical_drift_recall(y_true,y_pred,critical_mask),
        "false_safe_rate": false_safe_rate(y_true,y_pred,critical_mask,faithful_index=faithful_index),
    }
