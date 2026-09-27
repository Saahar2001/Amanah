from __future__ import annotations

import re
from amanah_engine.schemas import DriftFinding
from data.models import DriftLabel, Severity

NEGATION = re.compile(r"\b(no|not|never|cannot|can't|does not|do not|did not)\b", re.I)
MODALITY = re.compile(r"\b(may|might|can|could|should|must|shall)\b", re.I)
QUANTIFIER = re.compile(r"\b(some|all|every|each|none|many|few)\b", re.I)
CONDITION = re.compile(r"\b(if|unless|provided that|when)\b", re.I)


def _first(pattern: re.Pattern, text: str) -> str | None:
    m=pattern.search(text)
    return m.group(0) if m else None


def run_rules(reference_en: str, candidate_en: str) -> list[DriftFinding]:
    findings: list[DriftFinding] = []
    ref_neg, cand_neg = _first(NEGATION, reference_en), _first(NEGATION, candidate_en)
    if bool(ref_neg) != bool(cand_neg):
        findings.append(DriftFinding(label=DriftLabel.NEGATION_FLIP,severity=Severity.S3,confidence=1.0,source_span=ref_neg,target_span=cand_neg,evidence="Negation presence differs between trusted reference and candidate.",origin="rule"))

    ref_modal, cand_modal = _first(MODALITY, reference_en), _first(MODALITY, candidate_en)
    if ref_modal and cand_modal and ref_modal.lower() != cand_modal.lower():
        findings.append(DriftFinding(label=DriftLabel.MODALITY_SHIFT,severity=Severity.S3,confidence=0.98,source_span=ref_modal,target_span=cand_modal,evidence="Modal force changed.",origin="rule"))

    ref_quant, cand_quant = _first(QUANTIFIER, reference_en), _first(QUANTIFIER, candidate_en)
    if ref_quant and cand_quant and ref_quant.lower() != cand_quant.lower():
        findings.append(DriftFinding(label=DriftLabel.QUANTIFIER_CHANGE,severity=Severity.S3,confidence=0.98,source_span=ref_quant,target_span=cand_quant,evidence="Quantifier changed.",origin="rule"))

    ref_cond, cand_cond = _first(CONDITION, reference_en), _first(CONDITION, candidate_en)
    if ref_cond and not cand_cond:
        findings.append(DriftFinding(label=DriftLabel.CONDITION_LOSS,severity=Severity.S3,confidence=0.98,source_span=ref_cond,target_span=None,evidence="Conditional marker from trusted reference is absent.",origin="rule"))

    ref_tokens=reference_en.split(); cand_tokens=candidate_en.split()
    if len(ref_tokens) >= 8 and len(cand_tokens) / max(1,len(ref_tokens)) < 0.45:
        findings.append(DriftFinding(label=DriftLabel.OMISSION,severity=Severity.S2,confidence=0.85,evidence="Candidate is substantially shorter than trusted reference; review for omitted meaning.",origin="rule"))
    return findings
