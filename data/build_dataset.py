from __future__ import annotations

from data.models import AmanahSample, DriftLabel, ReferenceTranslation, Severity


def build_faithful_samples(*, ayah_id: str, surah: int, source_ar: str, references: list[ReferenceTranslation]) -> list[AmanahSample]:
    rows: list[AmanahSample] = []
    for idx, ref in enumerate(references, start=1):
        rows.append(
            AmanahSample(
                sample_id=f"{ayah_id.replace(':', '_')}_faithful_{idx}",
                ayah_id=ayah_id,
                surah=surah,
                source_ar=source_ar,
                references_en=references,
                candidate_en=ref.text,
                labels=[DriftLabel.FAITHFUL],
                severity=Severity.S0,
                origin="trusted_reference",
                metadata={
                    "reference_provider": ref.provider,
                    "reference_version": ref.version,
                    "reference_checksum": ref.checksum,
                },
            )
        )
    return rows
