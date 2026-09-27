from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class AmanahMultiTaskConfig:
    num_drift_labels: int
    num_severity_labels: int = 4
    dropout: float = 0.1
    drift_loss_weight: float = 1.0
    severity_loss_weight: float = 0.5
    drift_pos_weight: tuple[float, ...] | None = None


@dataclass
class AmanahMultiTaskOutput:
    drift_logits: torch.Tensor
    severity_logits: torch.Tensor
    loss: torch.Tensor | None = None


class AmanahMultiTaskModel(nn.Module):
    def __init__(self, encoder: nn.Module, config: AmanahMultiTaskConfig):
        super().__init__()
        self.encoder = encoder
        self.config = config
        hidden_size = int(encoder.config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.drift_head = nn.Linear(hidden_size, config.num_drift_labels)
        self.severity_head = nn.Linear(hidden_size, config.num_severity_labels)

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denom

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None,
                drift_labels: torch.Tensor | None = None,
                severity_labels: torch.Tensor | None = None, **encoder_kwargs) -> AmanahMultiTaskOutput:
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **encoder_kwargs)
        pooled = self.dropout(self._pool(encoded.last_hidden_state, attention_mask))
        drift_logits = self.drift_head(pooled)
        severity_logits = self.severity_head(pooled)
        loss = None
        losses = []
        if drift_labels is not None:
            pos_weight = None
            if self.config.drift_pos_weight is not None:
                pos_weight = torch.as_tensor(self.config.drift_pos_weight, dtype=drift_logits.dtype, device=drift_logits.device)
            losses.append(self.config.drift_loss_weight * nn.functional.binary_cross_entropy_with_logits(
                drift_logits, drift_labels.float(), pos_weight=pos_weight
            ))
        if severity_labels is not None:
            losses.append(self.config.severity_loss_weight * nn.functional.cross_entropy(severity_logits, severity_labels.long()))
        if losses:
            loss = torch.stack(losses).sum()
        return AmanahMultiTaskOutput(drift_logits=drift_logits, severity_logits=severity_logits, loss=loss)

    @classmethod
    def from_pretrained_encoder(cls, model_name: str, config: AmanahMultiTaskConfig) -> "AmanahMultiTaskModel":
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise RuntimeError("transformers is required for pretrained training; install requirements.txt") from exc
        return cls(AutoModel.from_pretrained(model_name), config)
