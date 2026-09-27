from __future__ import annotations

from dataclasses import dataclass
from amanah_engine.schemas import DriftFinding
from data.models import Decision, Severity


@dataclass(frozen=True)
class FusionResult:
    decision: Decision
    integrity_score: int
    severity: Severity
    confidence: float
    findings: list[DriftFinding]
    needs_human_review: bool
    notes: list[str]


def _max_severity(findings: list[DriftFinding]) -> Severity:
    rank={Severity.S0:0,Severity.S1:1,Severity.S2:2,Severity.S3:3}
    return max((f.severity for f in findings), key=lambda x: rank[x], default=Severity.S0)


def _score(findings: list[DriftFinding]) -> int:
    penalties={Severity.S0:0,Severity.S1:5,Severity.S2:20,Severity.S3:45}
    penalty=sum(round(penalties[f.severity]*f.confidence) for f in findings)
    return max(0,100-penalty)


def fuse_decision(*, model_findings: list[DriftFinding], rule_findings: list[DriftFinding], model_available: bool, model_confidence: float) -> FusionResult:
    all_findings=list(rule_findings)+list(model_findings)
    severity=_max_severity(all_findings)
    confidence=max([model_confidence]+[f.confidence for f in all_findings]) if all_findings else model_confidence
    if not model_available:
        return FusionResult(Decision.ABSTAIN,_score(all_findings),severity,confidence,all_findings,True,["Model unavailable; human review required."])
    if any(f.severity == Severity.S3 and f.origin == "rule" for f in rule_findings):
        return FusionResult(Decision.CRITICAL,_score(all_findings),Severity.S3,confidence,all_findings,True,["High-precision critical rule triggered."])
    if model_confidence < 0.45:
        return FusionResult(Decision.ABSTAIN,_score(all_findings),severity,confidence,all_findings,True,["Model confidence below abstention threshold."])
    if any(f.severity == Severity.S3 for f in all_findings):
        return FusionResult(Decision.CRITICAL,_score(all_findings),Severity.S3,confidence,all_findings,True,["Critical semantic drift detected."])
    if all_findings:
        return FusionResult(Decision.REVIEW,_score(all_findings),severity,confidence,all_findings,True,["Potential semantic drift requires review."])
    return FusionResult(Decision.PASS,100,Severity.S0,confidence,[],False,[])
