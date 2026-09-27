from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class DriftLabel(str, Enum):
    FAITHFUL = "FAITHFUL"
    NEGATION_FLIP = "NEGATION_FLIP"
    OMISSION = "OMISSION"
    ADDITION = "ADDITION"
    MODALITY_SHIFT = "MODALITY_SHIFT"
    QUANTIFIER_CHANGE = "QUANTIFIER_CHANGE"
    ENTITY_SWAP = "ENTITY_SWAP"
    AGENCY_SHIFT = "AGENCY_SHIFT"
    CONDITION_LOSS = "CONDITION_LOSS"
    TEMPORAL_SHIFT = "TEMPORAL_SHIFT"
    TERM_FLATTENING = "TERM_FLATTENING"
    LEXICAL_SEMANTIC_SHIFT = "LEXICAL_SEMANTIC_SHIFT"
    SEMANTIC_NARROWING = "SEMANTIC_NARROWING"
    SEMANTIC_BROADENING = "SEMANTIC_BROADENING"
    SEMANTIC_GRADATION_LOSS = "SEMANTIC_GRADATION_LOSS"
    INTERPRETATION_ADDITION = "INTERPRETATION_ADDITION"
    UNCERTAIN = "UNCERTAIN"


TRAINABLE_DRIFT_LABELS_V0 = [
    DriftLabel.FAITHFUL,
    DriftLabel.NEGATION_FLIP,
    DriftLabel.OMISSION,
    DriftLabel.MODALITY_SHIFT,
    DriftLabel.QUANTIFIER_CHANGE,
    DriftLabel.CONDITION_LOSS,
]


class Severity(str, Enum):
    S0 = "S0"
    S1 = "S1"
    S2 = "S2"
    S3 = "S3"


class Decision(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    CRITICAL = "CRITICAL"
    ABSTAIN = "ABSTAIN"


class ReferenceTranslation(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1)
    translator: str = Field(min_length=1)
    version: str = Field(min_length=1)
    text: str = Field(min_length=1)
    source_url: HttpUrl
    retrieved_at: str = Field(min_length=1)
    checksum: str = Field(min_length=1)


class AmanahSample(BaseModel):
    sample_id: str = Field(min_length=1)
    ayah_id: str = Field(pattern=r"^\d+:\d+$")
    surah: int = Field(ge=1, le=114)
    source_type: Literal["quran"] = "quran"
    source_ar: str = Field(min_length=1)
    references_en: list[ReferenceTranslation] = Field(min_length=1)
    candidate_en: str = Field(min_length=1)
    labels: list[DriftLabel] = Field(min_length=1)
    severity: Severity
    origin: Literal["trusted_reference", "synthetic_mutation", "human_authored"]
    mutation_type: DriftLabel | None = None
    changed_span: str | None = None
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    @field_validator("labels")
    @classmethod
    def faithful_is_exclusive(cls, labels: list[DriftLabel]) -> list[DriftLabel]:
        if DriftLabel.FAITHFUL in labels and len(labels) > 1:
            raise ValueError("FAITHFUL cannot coexist with drift labels")
        return labels
