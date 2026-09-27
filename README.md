# AMANAH ML Engine — أمانة

AMANAH is a **semantic-integrity QA layer** for Islamic-content translation. The v0.1 ML scope is deliberately narrow: **trusted Qur'anic Arabic → candidate English translation**.

The engine does not issue fatwas, does not certify a translation as religiously infallible, and does not modify the canonical Qur'anic source. It combines a custom fine-tuned multilingual classifier, deterministic high-precision rules, multiple trusted references, confidence/severity fusion, and human review.

## System flow

```text
Trusted Arabic source + candidate English
              ↓
Canonical source / provenance match
              ↓
Three-reference English envelope
              ↓
AMANAH custom drift classifier
              +
Deterministic critical rules
              ↓
Decision fusion
              ↓
PASS | REVIEW | CRITICAL | ABSTAIN
              ↓
Optional Cloudflare Llama explanation only
              ↓
Human review / publication workflow
```

## Safety boundary

- v0.1 is **Qur'an Arabic → English only**.
- Canonical Arabic and trusted translations are immutable inputs.
- Synthetic mutations are made only on candidate copies.
- Missing provenance, source mismatch, unavailable model, and low-confidence cases fail closed to `ABSTAIN`.
- `REVIEW`, `CRITICAL`, and `ABSTAIN` require human review.
- The downstream Llama explanation layer cannot overwrite AMANAH `decision`, `severity`, or `drifts`.
- A `PASS` result means **no material drift detected by this analysis**, not religious certification.

## Trusted source bundle

The automated source builder uses:

- **Tanzil Qur'an Text — Uthmani v1.1** for canonical Arabic, preserving the text verbatim and source attribution.
- **QuranEnc** English references discovered through the official API:
  - Rowwad Translation Center (`english_rwwad`);
  - Noor International Center (`english_saheeh`);
  - Hilali & Khan (`english_hilali_khan`).

The current versions are read from QuranEnc at retrieval time and stored with provenance instead of being silently hard-coded into the runtime data.

```bash
python scripts/fetch_verified_sources.py \
  --output data/private/reference_bundle.json \
  --runtime-store data/reference_store.json
```

The repository intentionally does **not** commit the full third-party reference corpus. Generated source bundles are gitignored.

## AMANAH-SD build + hard QA gate

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

The QA gate checks:

- 6,236 canonical ayat;
- exactly three references per ayah;
- duplicate/empty source records;
- canonical consistency;
- unknown ayah IDs;
- zero `ayah_id` leakage across train/validation/test;
- active-label coverage.

Active trainable v0 labels:

`FAITHFUL`, `NEGATION_FLIP`, `OMISSION`, `MODALITY_SHIFT`, `QUANTIFIER_CHANGE`, `CONDITION_LOSS`.

Other AMANAH taxonomy labels remain reserved until validated training data exists for them.

## Fastest real training path: Google Colab GPU

Use the notebook committed at:

`notebooks/AMANAH_Training_Colab.ipynb`

It runs this sequence end-to-end:

1. clones `https://github.com/Saahar2001/Amanah.git`;
2. installs dependencies and runs the repository tests;
3. asserts a CUDA GPU is available;
4. retrieves the verified source bundle;
5. builds and QA-checks AMANAH-SD;
6. fine-tunes multilingual mDeBERTa with class weighting, gradient clipping and validation-based checkpoint selection;
7. calibrates per-label thresholds on **validation only**;
8. evaluates exactly once on the frozen held-out test split;
9. packages a self-contained Hugging Face custom-handler model repo;
10. locally smoke-tests that packaged handler;
11. produces `AMANAH_FINAL_REPORT.json` and `AMANAH_HF_DEPLOYMENT.zip`;
12. persists both artifacts to `MyDrive/AMANAH_Artifacts` for later QA/review;
13. optionally uploads the measured model package to a private Hugging Face model repository when `HF_TOKEN` is present in Colab Secrets and records the model-repo URL in the final report.

Do **not** publish a metric until this run produces `artifacts/evaluation/metrics.json`.

## Training CLI

```bash
python -m training.train \
  --train artifacts/amanah_sd/train.jsonl \
  --validation artifacts/amanah_sd/validation.jsonl \
  --output-dir checkpoints/amanah-drift-v0.1 \
  --epochs 5 \
  --batch-size 4 \
  --gradient-accumulation 4 \
  --learning-rate 2e-5 \
  --max-length 256 \
  --patience 2
```

Then:

```bash
python -m training.calibrate_thresholds \
  --checkpoint checkpoints/amanah-drift-v0.1 \
  --validation artifacts/amanah_sd/validation.jsonl

python -m training.evaluate \
  --checkpoint checkpoints/amanah-drift-v0.1 \
  --test artifacts/amanah_sd/test.jsonl \
  --output artifacts/evaluation
```

Primary KPI: **Critical Drift Recall**. Also report **False Safe Rate**, Macro F1, per-label F1, and Severity Macro F1.

## Hugging Face packaging

```bash
python scripts/package_hf_model.py \
  --checkpoint checkpoints/amanah-drift-v0.1 \
  --reference-store data/reference_store.json \
  --output artifacts/hf_model_repo
```

The package includes the trained checkpoint, tokenizer, calibrated thresholds, reference store, model config, a root `handler.py`, custom dependencies, and model card metadata for a Hugging Face custom Inference Endpoint.

The dedicated endpoint itself is created **after** the model repo is uploaded because endpoint hardware/region is a billed Hugging Face resource and must be selected under the owner's account.

## FastAPI fallback

```bash
export MODEL_DIR=checkpoints/amanah-drift-v0.1
export SOURCE_REGISTRY_PATH=data/reference_store.json
export AMANAH_API_TOKEN='replace-me'
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Liveness:

```text
GET /health
```

Readiness — returns 503 until both model and references are actually loaded:

```text
GET /ready
```

Analyze:

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

## Existing Cloudflare site integration

Mahmoud should use `integration-cloudflare.ts` and follow `docs/MAHMOUD_HANDOFF.md`.

For the Hugging Face transport, Cloudflare sends:

```json
{
  "inputs": {
    "source_type": "quran",
    "source_ar": "...",
    "candidate_en": "...",
    "ayah_id": "2:256"
  }
}
```

If the ML endpoint is unavailable, the integration fails closed to `ABSTAIN` instead of falling back to a confident LLM verdict.

## QA

Local release gates:

```bash
pytest -q
python -m training.train --smoke
python -m compileall -q amanah_engine api data training scripts deployment
git diff --check
```

GitHub Actions repeats the unit/contract suite, training-architecture smoke, compile check, and live upstream source-contract smoke on every push/PR.

Full checklist: `docs/QA_CHECKLIST.md`.

## Repository map

```text
amanah_engine/          runtime classifier, references, rules, fusion/service
api/                    FastAPI fallback API
data/                   schemas, deterministic mutations, splits, validation
training/               multitask model, training, threshold calibration, metrics
deployment/             Hugging Face handler, Docker/Render fallback
scripts/                source retrieval, dataset build QA, HF packaging
notebooks/              one-click Colab GPU pipeline
examples/               minimal perturbation/demo inputs
tests/                  unit, contract, leakage, safety and deployment tests
docs/                   dataset/evaluation/integration/pitch/release documentation
integration-cloudflare.ts
```

## What can be claimed today

Before a real GPU run, say:

> AMANAH implements a hybrid semantic-integrity pipeline with a reproducible custom fine-tuning workflow, deterministic critical-drift rules, trusted-source provenance, validation-calibrated thresholds, fail-closed abstention, and human review.

After the frozen held-out test produces metrics, you may add the **measured** values from `AMANAH_FINAL_REPORT.json`.

Never invent or round an unmeasured accuracy claim.
