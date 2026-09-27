# AMANAH — Semantic Integrity Engine

AMANAH is a semantic-integrity quality-assurance layer for multilingual Islamic content. The current measured model scope is Qur’anic Arabic → English translation review.

The system is designed to detect whether protected elements of meaning changed during translation, identify the drift type and severity, attach evidence, and route uncertain or high-risk cases to human review.

## System Architecture

```text
Trusted Arabic Source
        +
Candidate English Translation
        ↓
Canonical Source Verification
        ↓
Trusted Multi-Reference Envelope
        ↓
Custom Semantic-Drift Classifier
        +
Deterministic Critical Rules
        ↓
Decision Fusion
        ↓
PASS | REVIEW | CRITICAL | ABSTAIN
        ↓
Evidence + Highlighted Findings
        ↓
Optional Explanation Layer
        ↓
Human Review
```

## Model Scope

The v0.1 classifier is restricted to:

- Qur’anic Arabic as the trusted canonical source.
- English candidate translations.
- Three trusted English reference translations with version and provenance tracking.
- Active drift classes:
  - `FAITHFUL`
  - `NEGATION_FLIP`
  - `OMISSION`
  - `MODALITY_SHIFT`
  - `QUANTIFIER_CHANGE`
  - `CONDITION_LOSS`

Reserved taxonomy classes remain in the domain schema and are not presented as trained outputs until validated training data exists for them.

## Data Governance

AMANAH separates source data into three immutable roles:

- `CANONICAL`: trusted Arabic Qur’anic text.
- `REFERENCE`: published trusted English translations.
- `CANDIDATE`: the translation being evaluated or a clearly labeled synthetic mutation.

Synthetic semantic mutations are never written back to canonical or reference content.

The source pipeline records source provider, translator/publisher, version, retrieval time, URL, and checksum for traceability.

## Training and Evaluation

The training pipeline implements:

- group splitting by `ayah_id` to prevent leakage;
- controlled semantic mutation generation;
- mutation validation gates;
- class-aware training;
- validation-loss checkpoint selection;
- gradient clipping;
- per-label threshold calibration on validation data only;
- frozen held-out test evaluation.

Primary evaluation priority:

**Critical Drift Recall**

Supporting metrics:

- False Safe Rate
- Macro F1
- per-label F1
- Severity Macro F1

No performance value is treated as valid until it is produced by the held-out evaluation pipeline.

## Runtime Safety

The runtime fails closed in the following conditions:

- trusted source not found;
- Arabic source mismatch;
- reference provenance unavailable;
- model unavailable;
- insufficient confidence.

These states return `ABSTAIN` and require human review.

A `PASS` result means that no material drift was detected by this analysis. It is not a religious certification, a fatwa, or a substitute for qualified human review.

## Deployment

The repository includes:

- a custom Hugging Face inference handler;
- FastAPI fallback service;
- Docker and Render deployment configuration;
- Cloudflare integration helper;
- readiness and liveness contracts;
- fail-closed transport behavior.

The platform integration keeps model outputs authoritative. The explanation layer may clarify a finding or suggest review wording, but it cannot overwrite the structured decision, severity, or detected drift labels.

## Repository Structure

```text
amanah_engine/   runtime classifier, references, rules, fusion
api/             FastAPI service
data/            schemas, mutations, split and validation utilities
training/        model, fine-tuning, calibration and evaluation
deployment/      inference handler and container configuration
scripts/         source retrieval, dataset QA and deployment packaging
notebooks/       measured training and validation pipeline
examples/        semantic perturbation benchmark examples
tests/           unit, contract, safety and deployment validation
docs/            architecture, QA, integration and presentation guidance
```

## Quality Gates

The main CI pipeline validates deterministic repository behavior on every update:

- unit and contract tests;
- model architecture smoke test;
- Python compilation;
- deployment and fail-closed contracts.

External source availability, full dataset retrieval, GPU training, held-out evaluation, and production endpoint deployment are validated in their respective execution environments.

## Integration Contract

The model service returns a stable structured response containing:

- decision;
- integrity score;
- severity;
- confidence;
- typed drift findings;
- human-review requirement;
- model version;
- reference verification status.

Platform integration details are documented in:

`docs/PLATFORM_INTEGRATION_GUIDE.md`

Presentation wording and claim boundaries are documented in:

`docs/PRESENTATION_REVISION_GUIDE.md`
