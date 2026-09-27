from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class SourceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    translator: str = Field(min_length=1)
    language: str = Field(min_length=2)
    version: str = Field(min_length=1)
    source_url: HttpUrl
    retrieved_at: str = Field(min_length=1)
    checksum: str = Field(min_length=1)
    status: Literal["approved_reference", "auxiliary", "experimental"]

    @field_validator("source_url")
    @classmethod
    def https_only(cls, value: HttpUrl) -> HttpUrl:
        if value.scheme != "https":
            raise ValueError("source_url must use https")
        return value


def verify_source_record(record: SourceRecord) -> bool:
    for field in ("source_id", "provider", "translator", "version", "retrieved_at", "checksum"):
        if not str(getattr(record, field)).strip():
            raise ValueError(f"missing {field}")
    if not record.checksum.startswith("sha256:"):
        raise ValueError("checksum must use sha256 prefix")
    return True


def load_source_registry(path: str | Path) -> list[SourceRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = [SourceRecord.model_validate(item) for item in payload]
    for record in records:
        verify_source_record(record)
    return records
