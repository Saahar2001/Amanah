from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.models import DriftLabel, Severity, TRAINABLE_DRIFT_LABELS_V0
from training.metrics import compute_drift_metrics


def write_evaluation_report(metrics: dict, output_dir: str | Path) -> tuple[Path, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "metrics.json"
    md_path = out / "EVALUATION.md"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    lines = ["# AMANAH Evaluation", "", "> Generated only from measured inputs; no metric is fabricated.", ""]
    for key, value in metrics.items():
        if isinstance(value, (str, int, float)):
            lines.append(f"- **{key}:** {value}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _finding_label(finding) -> str:
    if isinstance(finding, dict):
        value = finding.get("label")
    else:
        value = getattr(finding, "label", None)
    return getattr(value, "value", value)


def evaluate_rows(rows: list[dict], predictor, label_names: list[str] | None = None) -> dict:
    if label_names is None:
        cfg = getattr(predictor, "_config", None)
        label_names = list(cfg.get("drift_labels", [])) if isinstance(cfg, dict) else []
    if not label_names:
        label_names = [x.value for x in TRAINABLE_DRIFT_LABELS_V0]
    labels = [DriftLabel(name) for name in label_names]
    label_index = {label.value: i for i, label in enumerate(labels)}
    sev_index = {severity.value: i for i, severity in enumerate(Severity)}
    y_true = np.zeros((len(rows), len(labels)), dtype=int)
    y_pred = np.zeros_like(y_true)
    severity_true = np.zeros(len(rows), dtype=int)
    severity_pred = np.zeros(len(rows), dtype=int)

    for row_idx, row in enumerate(rows):
        for label in row["labels"]:
            y_true[row_idx, label_index[label]] = 1
        severity_true[row_idx] = sev_index[row["severity"]]
        result = predictor.predict(row["source_ar"], row["candidate_en"])
        if not result.get("available"):
            return {"status": "model_unavailable", "rows": row_idx, "model_version": result.get("model_version", "unknown"), "error": result.get("error", "model unavailable")}
        findings = result.get("findings", [])
        if findings:
            for finding in findings:
                label = _finding_label(finding)
                if label in label_index:
                    y_pred[row_idx, label_index[label]] = 1
        else:
            y_pred[row_idx, label_index[DriftLabel.FAITHFUL.value]] = 1
        predicted_severity = result.get("severity")
        if predicted_severity not in sev_index:
            predicted_severity = "S0" if not findings else max((getattr(getattr(f, "severity", None), "value", None) or (f.get("severity") if isinstance(f, dict) else "S0") for f in findings), key=lambda s: sev_index.get(s, 0), default="S0")
        severity_pred[row_idx] = sev_index[predicted_severity]

    metrics = compute_drift_metrics(y_true,y_pred,severity_true,severity_pred,critical_mask=(severity_true == sev_index[Severity.S3.value]),faithful_index=label_index.get(DriftLabel.FAITHFUL.value))
    return {"status": "ok", "rows": len(rows), "model_version": getattr(predictor, "model_version", "unknown"), **metrics}


def evaluate_checkpoint(*, checkpoint: str, test_path: str, output_dir: str) -> dict:
    from amanah_engine.classifier import ClassifierAdapter
    test = Path(test_path)
    if not test.exists(): raise FileNotFoundError(test)
    rows = [json.loads(line) for line in test.read_text(encoding="utf-8").splitlines() if line.strip()]
    predictor = ClassifierAdapter(checkpoint)
    metrics = evaluate_rows(rows, predictor, label_names=list(predictor._config.get("drift_labels", [])) if getattr(predictor, "_config", None) else None)
    write_evaluation_report(metrics, output_dir)
    return metrics


def main():
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", required=True); p.add_argument("--test", required=True); p.add_argument("--output", required=True)
    a = p.parse_args(); print(json.dumps(evaluate_checkpoint(checkpoint=a.checkpoint, test_path=a.test, output_dir=a.output)))


if __name__ == "__main__": main()
