from __future__ import annotations

import re
from dataclasses import dataclass

from data.models import AmanahSample, DriftLabel
from data.mutations import MutationResult


@dataclass(frozen=True)
class ValidationResult:
    valid: bool
    reasons: list[str]


def _contains_negation(text: str) -> bool:
    return bool(re.search(r"\b(no|not|never|cannot|can't|does not|do not|did not)\b", text, re.I))


def validate_mutation(sample: AmanahSample, mutation: MutationResult, *, expected_source_ar: str | None = None) -> ValidationResult:
    reasons: list[str] = []
    if expected_source_ar is not None and sample.source_ar != expected_source_ar:
        reasons.append("canonical source modified")
    if not mutation.candidate.strip():
        reasons.append("empty candidate")
    if mutation.candidate.strip() == mutation.original.strip():
        reasons.append("candidate unchanged")
    if mutation.original != sample.candidate_en:
        reasons.append("mutation original does not match candidate")
    if int(mutation.metadata.get("operations", 1)) > 1:
        reasons.insert(0, "multiple unintended operations detected")

    if mutation.label == DriftLabel.NEGATION_FLIP:
        span_is_negation = bool(mutation.changed_span and _contains_negation(mutation.changed_span))
        if not _contains_negation(mutation.original) or _contains_negation(mutation.candidate) or not span_is_negation:
            reasons.append("negation label mismatch")
    elif mutation.label == DriftLabel.QUANTIFIER_CHANGE:
        if not re.search(r"\b(some|all|every|none)\b", mutation.original, re.I):
            reasons.append("quantifier label mismatch")
    elif mutation.label == DriftLabel.MODALITY_SHIFT:
        if not re.search(r"\b(may|can|should|must)\b", mutation.original, re.I):
            reasons.append("modality label mismatch")
    elif mutation.label == DriftLabel.CONDITION_LOSS:
        if not re.search(r"\bif\b", mutation.original, re.I) or re.search(r"\bif\b", mutation.candidate, re.I):
            reasons.append("condition label mismatch")
    elif mutation.label == DriftLabel.OMISSION:
        if len(mutation.candidate) >= len(mutation.original):
            reasons.append("omission label mismatch")
    elif mutation.label not in DriftLabel:
        reasons.append("unknown mutation label")

    # Conservative sanity check: candidates should retain at least 35% of token count unless condition loss.
    original_tokens = mutation.original.split()
    candidate_tokens = mutation.candidate.split()
    if original_tokens and mutation.label != DriftLabel.CONDITION_LOSS and len(candidate_tokens) / len(original_tokens) < 0.35:
        reasons.append("candidate excessively truncated")

    return ValidationResult(valid=not reasons, reasons=reasons)
