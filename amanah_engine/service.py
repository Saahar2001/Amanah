from __future__ import annotations

from amanah_engine.references import normalize_arabic
from amanah_engine.schemas import AnalyzeRequest, AnalyzeResponse, DriftFinding
from amanah_engine.semantic_rules import run_rules
from amanah_engine.scoring import fuse_decision
from data.models import Decision, Severity


def _consensus_rule_findings(references_en: list[str], candidate_en: str) -> list[DriftFinding]:
    if not references_en:
        return []
    per_reference = [run_rules(str(ref), candidate_en) for ref in references_en]
    common_labels = set(f.label for f in per_reference[0])
    for findings in per_reference[1:]:
        common_labels &= {f.label for f in findings}
    return [f for f in per_reference[0] if f.label in common_labels]


class AmanahAnalysisService:
    def __init__(self, classifier, reference_store):
        self.classifier=classifier; self.reference_store=reference_store

    def readiness(self) -> dict:
        model_available = bool(getattr(self.classifier, "available", False))
        try:
            reference_count = len(self.reference_store)
        except (TypeError, AttributeError):
            reference_count = 0
        return {
            "ready": model_available and reference_count > 0,
            "model_available": model_available,
            "reference_count": reference_count,
        }

    def _abstain(self, model_version: str, note: str, reference_status: str) -> AnalyzeResponse:
        return AnalyzeResponse(decision=Decision.ABSTAIN,integrity_score=0,severity=Severity.S0,confidence=0.0,drifts=[],needs_human_review=True,model_version=model_version,reference_status=reference_status,notes=[note])

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        ayah_id = request.ayah_id or (f'{request.surah}:{request.ayah}' if request.surah and request.ayah else None)
        reference=self.reference_store.lookup(ayah_id)
        if reference is None and hasattr(self.reference_store, 'match_source_ar'):
            reference=self.reference_store.match_source_ar(request.source_ar)
        if reference is None:
            return self._abstain(getattr(self.classifier,'model_version','unknown'),"Trusted reference not found; provenance cannot be verified.","missing")
        if reference.get('provenance_verified') is not True:
            return self._abstain(getattr(self.classifier,'model_version','unknown'),"Trusted reference provenance is not verified.","unverified_provenance")
        canonical_ar = reference.get('source_ar')
        if canonical_ar and normalize_arabic(request.source_ar) != normalize_arabic(str(canonical_ar)):
            return self._abstain(getattr(self.classifier,'model_version','unknown'),"Submitted Arabic source does not match the trusted canonical source for this ayah.","source_mismatch")
        prediction=self.classifier.predict(request.source_ar,request.candidate_en)
        if not prediction.get('available'):
            return self._abstain(prediction.get('model_version','unavailable'),prediction.get('error','Model unavailable.'),"verified")
        refs=reference.get('references_en') or []
        if not refs:
            return self._abstain(prediction.get('model_version','unknown'),"Trusted English reference missing.","missing")
        # Reference-envelope rule findings must survive all approved references before they are treated as high-precision.
        rule_findings=_consensus_rule_findings([str(ref) for ref in refs],request.candidate_en)
        raw_findings=prediction.get('findings',[])
        model_findings=[x if isinstance(x,DriftFinding) else DriftFinding.model_validate(x) for x in raw_findings]
        fused=fuse_decision(model_findings=model_findings,rule_findings=rule_findings,model_available=True,model_confidence=float(prediction.get('confidence',0.0)))
        return AnalyzeResponse(decision=fused.decision,integrity_score=fused.integrity_score,severity=fused.severity,confidence=fused.confidence,drifts=fused.findings,needs_human_review=fused.needs_human_review,model_version=prediction.get('model_version','unknown'),reference_status='verified',notes=fused.notes)
