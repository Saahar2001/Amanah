from __future__ import annotations

import re
from dataclasses import dataclass, field

from data.models import AmanahSample, DriftLabel


@dataclass(frozen=True)
class MutationResult:
    original: str
    candidate: str
    label: DriftLabel
    changed_span: str | None
    metadata: dict[str, object] = field(default_factory=dict)


def _replace_once(text: str, pattern: str, repl: str) -> tuple[str, str] | None:
    m = re.search(pattern, text, flags=re.IGNORECASE)
    if not m:
        return None
    return text[:m.start()] + repl + text[m.end():], m.group(0)


def apply_negation_flip(sample: AmanahSample) -> MutationResult | None:
    text = sample.candidate_en
    replacements = [
        (r"\bdoes\s+not\b", "does"),
        (r"\bdo\s+not\b", "do"),
        (r"\bdid\s+not\b", "did"),
        (r"\bcannot\b", "can"),
        (r"\bcan't\b", "can"),
        (r"\bnot\b", ""),
        (r"\bno\b", ""),
        (r"\bnever\b", ""),
    ]
    for pat, repl in replacements:
        out = _replace_once(text, pat, repl)
        if out:
            candidate, span = out
            candidate = re.sub(r"\s{2,}", " ", candidate).strip()
            candidate = re.sub(r"\s+([,.!?;:])", r"\1", candidate)
            return MutationResult(text, candidate, DriftLabel.NEGATION_FLIP, span, {"operations": 1})
    return None


def apply_quantifier_change(sample: AmanahSample) -> MutationResult | None:
    pairs = [(r"\bsome\b", "all"), (r"\ball\b", "some"), (r"\bevery\b", "some"), (r"\bnone\b", "all")]
    for pat, repl in pairs:
        out = _replace_once(sample.candidate_en, pat, repl)
        if out:
            candidate, span = out
            return MutationResult(sample.candidate_en, candidate, DriftLabel.QUANTIFIER_CHANGE, span, {"operations": 1})
    return None


def apply_modality_shift(sample: AmanahSample) -> MutationResult | None:
    pairs = [(r"\bmay\b", "must"), (r"\bcan\b", "must"), (r"\bshould\b", "must"), (r"\bmust\b", "may")]
    for pat, repl in pairs:
        out = _replace_once(sample.candidate_en, pat, repl)
        if out:
            candidate, span = out
            return MutationResult(sample.candidate_en, candidate, DriftLabel.MODALITY_SHIFT, span, {"operations": 1})
    return None


def apply_omission(sample: AmanahSample) -> MutationResult | None:
    text = sample.candidate_en
    # Only remove a trailing coordinated clause when the boundary is obvious.
    m = re.search(r",\s+(and|but)\s+.+?[.!?]?$", text, flags=re.IGNORECASE)
    if not m:
        return None
    candidate = (text[:m.start()].rstrip() + ".").replace("..", ".")
    return MutationResult(text, candidate, DriftLabel.OMISSION, m.group(0), {"operations": 1})


def apply_condition_loss(sample: AmanahSample) -> MutationResult | None:
    text = sample.candidate_en
    m = re.match(r"\s*if\s+(.+?),\s*(?:then\s+)?(.+)$", text, flags=re.IGNORECASE)
    if not m:
        return None
    candidate = m.group(2).strip()
    if candidate and candidate[0].islower():
        candidate = candidate[0].upper() + candidate[1:]
    return MutationResult(text, candidate, DriftLabel.CONDITION_LOSS, text[:m.start(2)], {"operations": 1})
