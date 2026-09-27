# AMANAH ML Engine — أمانة

AMANAH is a semantic-integrity quality-assurance layer for Islamic-content translation. The v0.1 measured scope is **trusted Qur'anic Arabic → candidate English translation**.

The engine combines trusted-source matching, a multi-reference envelope, a fine-tuned multilingual classifier, deterministic high-precision rules, calibrated confidence thresholds, decision fusion, and human review.

## Processing flow

```text
Trusted Arabic source + candidate English
              ↓
Canonical source / provenance match
              ↓
Three-reference English envelope
              ↓
Custom drift classifier + deterministic rules
              ↓
Decision fusion
              ↓
PASS | REVIEW | CRITICAL | ABSTAIN
              ↓
Optional LLM explanation
              ↓
Human review / publication workflow
```

## Scope and safety

- v0.1 is Qur'an Arabic → English only.
- Canonical Arabic and trusted references are immutable.
- Synthetic mutations are generated only as candidate copies.
- Missing provenance, source mismatch, model unavailability, or low confidence fail closed to `ABSTAIN`.
- `REVIEW`, `CRITICAL`, and `ABSTAIN` require human review.
- The explanation layer cannot overwrite `decision`, `severity`, or `drifts`.
- `PASS` means no material drift was detected by this analysis; it is not a religious certification.

## Source bundle

The source pipeline uses:

- Tanzil Qur'an Text — Uthmani v1.1 for canonical Arabic.
- Three QuranEnc English references discovered from the official API:
  - Rowwad Translation Center;
  - Noor International Center;
  - Hilali & Khan.

Versions, URLs, retrieval time, and checksums are stored at retrieval time.

```bash
python scripts/fetch_verified_sources.py \
  --output data/private/reference_bundle.json \
  --runtime-store data/reference_store.json
```

## Dataset build and QA

```bash
python -m scripts.prepare_amanah_sd \
  data/private/reference_bundle.json \
  artifacts/amanah_sd \
  --seed 42

python scripts/qa_dataset.py \
  --reference-bundle data/private/reference_bundle.json \
  --split-dir artifacts/amanah_sd \
  --output artifacts/amanah_sd/qa_report.json
```

The QA gate checks canonical count, reference count, duplicate/empty source records, canonical consistency, unknown ayah IDs, zero ayah leakage across splits, and active-label coverage.

Active v0 labels:

`FAITHFUL`, `NEGATION_FLIP`, `OMISSION`, `MODALITY_SHIFT`, `QUANTIFIER_CHANGE`, `CONDITION_LOSS`.

## Training notebook

Use:

`notebooks/model_training_pipeline.ipynb`

Before running, add two Colab Secrets:

- `REPOSITORY_URL` — the Git clone URL for this repository.
- `HF_TOKEN` — optional Hugging Face write token for private model upload.

The notebook performs source validation, dataset QA, GPU fine-tuning, validation-only threshold calibration, frozen test evaluation, deployment packaging, local handler smoke testing, and optional Hugging Face upload.

Do not publish any model metric until `artifacts/evaluation/metrics.json` exists.

## API

```http
POST /v1/analyze
Authorization: Bearer <token>
Content-Type: application/json
```

```json
{
  "source_type": "quran",
  "source_ar": "...",
  "candidate_en": "...",
  "ayah_id": "2:256"
}
```

Readiness:

```text
GET /ready
```

This returns 503 until the trained model and reference store are both loaded.

## Platform integration

See `docs/PLATFORM_INTEGRATION_GUIDE.md`.

## Quality assurance

Main CI runs deterministic unit/contract checks only. Live upstream source checks are executed from the training notebook or manually, so temporary third-party network failures do not make the core build red.

```bash
pytest -q
python -m training.train --smoke
python -m compileall -q amanah_engine api data training scripts deployment
git diff --check
```

See `docs/QUALITY_ASSURANCE_CHECKLIST.md`.

## Repository structure

```text
amanah_engine/   runtime classifier, references, rules, fusion
api/             FastAPI fallback service
data/            schemas, mutations, split and validation utilities
training/        model, training, calibration and evaluation
deployment/      Hugging Face handler and container fallback
scripts/         source retrieval, dataset QA and packaging
notebooks/       training pipeline
examples/        benchmark examples
tests/           unit and contract tests
docs/            technical and delivery documentation
```
