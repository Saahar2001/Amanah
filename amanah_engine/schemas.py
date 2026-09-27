from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from data.models import Decision, DriftLabel, Severity


class AnalyzeRequest(BaseModel):
    source_type: Literal["quran"]
    source_ar: str = Field(min_length=1)
    candidate_en: str = Field(min_length=1)
    ayah_id: str | None = Field(default=None, pattern=r"^\d+:\d+$")
    surah: int | None = Field(default=None, ge=1, le=114)
    ayah: int | None = Field(default=None, ge=1)


class DriftFinding(BaseModel):
    label: DriftLabel
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    source_span: str | None = None
    target_span: str | None = None
    evidence: str | None = None
    origin: Literal["model", "rule", "fusion"] = "model"


class AnalyzeResponse(BaseModel):
    decision: Decision
    integrity_score: int = Field(ge=0, le=100)
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    drifts: list[DriftFinding] = Field(default_factory=list)
    needs_human_review: bool
    model_version: str
    reference_status: str = "unknown"
    notes: list[str] = Field(default_factory=list)
