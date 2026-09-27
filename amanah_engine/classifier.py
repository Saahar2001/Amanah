from __future__ import annotations

import json
from pathlib import Path

from amanah_engine.schemas import DriftFinding
from data.models import DriftLabel, Severity


def build_encoder_from_saved_config(config_dict: dict):
    try:
        from transformers import AutoConfig, AutoModel
    except ImportError as exc:
        raise RuntimeError("transformers is required to load AMANAH checkpoints") from exc
    payload = dict(config_dict)
    model_type = payload.pop("model_type", None)
    if not model_type:
        raise ValueError("saved encoder config is missing model_type")
    hf_config = AutoConfig.for_model(model_type, **payload)
    return AutoModel.from_config(hf_config)


class ClassifierAdapter:
    def __init__(self, model_dir: str | Path, thresholds: list[float] | None = None):
        self.model_dir=Path(model_dir)
        self.thresholds=thresholds
        self.available=False
        self.model_version=self.model_dir.name or "unavailable"
        self.error: str | None=None
        self._model=None; self._tokenizer=None; self._config=None
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoTokenizer
            from training.modeling import AmanahMultiTaskConfig, AmanahMultiTaskModel
            cfg=json.loads((self.model_dir/'amanah_config.json').read_text(encoding='utf-8'))
            encoder_config = cfg.get('encoder_config')
            if encoder_config:
                encoder = build_encoder_from_saved_config(encoder_config)
            else:
                from transformers import AutoModel
                encoder = AutoModel.from_pretrained(cfg['base_model'])
            pos_weight = cfg.get('drift_pos_weight')
            model=AmanahMultiTaskModel(encoder,AmanahMultiTaskConfig(
                num_drift_labels=len(cfg['drift_labels']),
                num_severity_labels=len(cfg.get('severity_labels', ['S0','S1','S2','S3'])),
                drift_pos_weight=tuple(pos_weight) if pos_weight else None,
            ))
            state=torch.load(self.model_dir/'model.pt',map_location='cpu')
            model.load_state_dict(state); model.eval()
            self._tokenizer=AutoTokenizer.from_pretrained(self.model_dir)
            self._model=model; self._config=cfg
            if self.thresholds is None:
                tpath=self.model_dir/'thresholds.json'
                self.thresholds=json.loads(tpath.read_text()) if tpath.exists() else [0.5]*len(cfg['drift_labels'])
            self.available=True
        except Exception as exc:
            self.available=False; self.error=f"{type(exc).__name__}: {exc}"

    def predict_scores(self, source_ar: str, candidate_en: str) -> dict:
        if not self.available:
            return {"available":False,"probabilities":[],"severity_probabilities":[],"model_version":self.model_version,"error":self.error or "model unavailable"}
        import torch
        max_length = int((self._config or {}).get('max_length', 256))
        tok=self._tokenizer(source_ar,candidate_en,return_tensors='pt',truncation=True,max_length=max_length)
        with torch.no_grad(): out=self._model(**tok)
        probs=torch.sigmoid(out.drift_logits)[0].cpu().tolist()
        sev_probs=torch.softmax(out.severity_logits,dim=-1)[0].cpu().tolist()
        return {"available":True,"probabilities":[float(x) for x in probs],"severity_probabilities":[float(x) for x in sev_probs],"model_version":self.model_version}

    def predict(self, source_ar: str, candidate_en: str) -> dict:
        scores=self.predict_scores(source_ar,candidate_en)
        if not scores.get("available"):
            return {"available":False,"confidence":0.0,"severity":"S0","findings":[],"model_version":self.model_version,"error":scores.get("error","model unavailable")}
        probs=scores["probabilities"]; sev_probs=scores["severity_probabilities"]
        labels=self._config['drift_labels']; severity_values=self._config['severity_labels']
        severity=Severity(severity_values[max(range(len(sev_probs)),key=sev_probs.__getitem__)])
        findings=[]
        for i,(name,p) in enumerate(zip(labels,probs)):
            if name == DriftLabel.FAITHFUL.value: continue
            if p >= self.thresholds[i]:
                findings.append(DriftFinding(label=DriftLabel(name),severity=severity,confidence=float(p),origin='model'))
        faithful_idx=labels.index(DriftLabel.FAITHFUL.value) if DriftLabel.FAITHFUL.value in labels else None
        faithful_prob=probs[faithful_idx] if faithful_idx is not None else 0.0
        confidence=float(max([faithful_prob]+[f.confidence for f in findings]))
        return {"available":True,"confidence":confidence,"severity":severity.value,"findings":findings,"model_version":self.model_version}
