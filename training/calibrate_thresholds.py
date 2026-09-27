from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


def calibrate_label_thresholds(probabilities, truth, candidates=(0.3,0.4,0.5,0.6,0.7)) -> list[float]:
    probs=np.asarray(probabilities,float); y=np.asarray(truth,int)
    if probs.shape != y.shape:
        raise ValueError("probabilities and truth must have the same shape")
    thresholds=[]
    for i in range(probs.shape[1]):
        best_score=-1.0; best_threshold=float(candidates[0])
        for threshold in candidates:
            pred=(probs[:,i] >= threshold).astype(int)
            score=float(f1_score(y[:,i],pred,zero_division=0))
            if score > best_score or (score == best_score and threshold > best_threshold):
                best_score=score; best_threshold=float(threshold)
        thresholds.append(best_threshold)
    return thresholds


def calibrate_predictor(rows: list[dict], predictor, label_names: list[str], output_path: str | Path,
                        candidates=(0.3,0.4,0.5,0.6,0.7)) -> list[float]:
    label_index={name:i for i,name in enumerate(label_names)}
    probs=[]; truth=[]
    for row in rows:
        result=predictor.predict_scores(row['source_ar'],row['candidate_en'])
        if not result.get('available'):
            raise RuntimeError(result.get('error','model unavailable during calibration'))
        probabilities=result['probabilities']
        if len(probabilities) != len(label_names):
            raise ValueError('predictor probability count does not match label_names')
        target=[0]*len(label_names)
        for label in row['labels']:
            if label not in label_index:
                raise ValueError(f'unknown label in validation data: {label}')
            target[label_index[label]]=1
        probs.append(probabilities); truth.append(target)
    thresholds=calibrate_label_thresholds(np.asarray(probs),np.asarray(truth),candidates=candidates)
    Path(output_path).write_text(json.dumps(thresholds,indent=2),encoding='utf-8')
    return thresholds


def main() -> None:
    from amanah_engine.classifier import ClassifierAdapter
    p=argparse.ArgumentParser(description='Calibrate AMANAH per-label thresholds on validation JSONL')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--validation',required=True)
    p.add_argument('--output',default=None)
    args=p.parse_args()
    rows=[json.loads(line) for line in Path(args.validation).read_text(encoding='utf-8').splitlines() if line.strip()]
    predictor=ClassifierAdapter(args.checkpoint)
    if not predictor.available:
        raise RuntimeError(predictor.error or 'model unavailable')
    label_names=predictor._config['drift_labels']
    output=Path(args.output) if args.output else Path(args.checkpoint)/'thresholds.json'
    print(json.dumps(calibrate_predictor(rows,predictor,label_names,output),indent=2))


if __name__=='__main__':
    main()
