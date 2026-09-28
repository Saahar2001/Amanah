---
library_name: transformers
pipeline_tag: text-classification
language:
  - ar
  - en
tags:
  - endpoints-template
  - semantic-integrity
  - multilingual
  - quran
  - semantic-drift
metrics:
  - f1
model-index:
  - name: Semantic Integrity Classifier v0.1
    results:
      - task:
          type: text-classification
          name: Semantic Drift Classification
        dataset:
          name: AMANAH-SD v0 held-out test
          type: amanah-sd-v0
        metrics:
          - type: f1
            name: Macro F1
            value: 0.9215299048367256
          - type: f1
            name: Severity Macro F1
            value: 0.9375443793332715
          - type: critical_drift_recall
            name: Critical Drift Recall
            value: 0.9441233140655106
          - type: false_safe_rate
            name: False Safe Rate
            value: 0.051059730250481696
---

# Semantic Integrity Classifier v0.1

A multilingual semantic-integrity classifier for detecting typed semantic drift between a trusted Arabic Qur'anic source and a candidate English translation.

The model is one component of a hybrid verification system. It is designed to support translation quality assurance and human review; it is not a religious authority and does not issue rulings.

## Measured held-out results

Evaluation was performed on a frozen held-out test split grouped by `ayah_id`, preventing the same ayah from appearing across train, validation, and test.

| Metric | Measured result |
|---|---:|
| **Critical Drift Recall** | **94.41%** |
| **Severity Macro F1** | **93.75%** |
| **Macro F1** | **92.15%** |
| **False Safe Rate** ↓ | **5.11%** |
| Held-out test samples | 3,323 |
| Held-out ayahs | 625 |

The primary safety-oriented KPI is **Critical Drift Recall**. **False Safe Rate is lower-is-better** and measures critical cases for which no drift was predicted.

## Per-label F1

| Trainable label | F1 |
|---|---:|
| FAITHFUL | 95.39% |
| NEGATION_FLIP | 92.95% |
| OMISSION | 91.64% |
| MODALITY_SHIFT | 96.18% |
| QUANTIFIER_CHANGE | 94.40% |
| CONDITION_LOSS | 82.35% |

`CONDITION_LOSS` currently has far fewer generated examples than the other labels, so expanding and independently reviewing this class is a priority for the next dataset version.

## Dataset snapshot

AMANAH-SD v0 uses immutable trusted-source/reference data and controlled semantic mutations.

| Dataset item | Count |
|---|---:|
| Qur'anic ayahs | 6,236 |
| Trusted English references per ayah | 3 |
| Total samples | 32,621 |
| Faithful reference samples | 18,708 |
| Controlled synthetic mutations | 13,913 |
| Rejected mutations during validation | 2,660 |
| Train samples | 25,986 |
| Validation samples | 3,312 |
| Test samples | 3,323 |

### Label distribution

| Label | Samples |
|---|---:|
| FAITHFUL | 18,708 |
| NEGATION_FLIP | 4,692 |
| OMISSION | 4,028 |
| QUANTIFIER_CHANGE | 3,293 |
| MODALITY_SHIFT | 1,769 |
| CONDITION_LOSS | 131 |

Dataset QA reported no split ayah leakage, canonical mismatches, unknown ayahs, duplicate ayah IDs, or missing trainable labels.

## Training summary

- **Base encoder:** `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`
- **Task:** multi-label semantic drift + four-class severity prediction
- **Seed:** 42
- **Maximum length:** 256
- **Target epochs:** 5
- **Early stopping:** after epoch 4
- **Best validation loss:** 0.22957
- **Calibration:** per-label thresholds selected on the validation split
- **Final evaluation:** frozen held-out test only

## Active v0.1 labels

- `FAITHFUL`
- `NEGATION_FLIP`
- `OMISSION`
- `MODALITY_SHIFT`
- `QUANTIFIER_CHANGE`
- `CONDITION_LOSS`

Other taxonomy labels remain reserved for later validated datasets and must not be presented as trained v0.1 outputs.

## System decision layer

The classifier feeds a deterministic verification layer that can return:

- `PASS`
- `REVIEW`
- `CRITICAL`
- `ABSTAIN`

Confidence and severity are separate signals. Low-confidence, unavailable-reference, and model-unavailable cases should fail closed to `ABSTAIN` and human review.

## Intended use

- pre-publication translation quality assurance;
- prioritizing passages for human review;
- detecting high-impact semantic changes;
- institutional integration through a structured API;
- research and controlled benchmarking of semantic drift.

## Not intended for

- issuing fatwas or religious rulings;
- replacing qualified translators or reviewers;
- claiming that any translation is infallible;
- modifying canonical Qur'anic text;
- evaluating unsupported languages or domains without separate validation.

## Important limitations

- The drift portion of AMANAH-SD v0 is based on controlled synthetic mutations; naturally occurring translation errors require a separately human-reviewed benchmark.
- Label distribution is imbalanced, especially for `CONDITION_LOSS`.
- Span evidence in v0.1 is rule/alignment based rather than a trained token-classification head.
- Multiple trusted translations may legitimately differ; disagreement is a review signal, not proof that one translation is uniquely correct.
- Reported metrics apply to the measured v0.1 held-out benchmark only.

## Reproducibility

The training, data QA, calibration, evaluation, packaging, and deployment code is maintained in the project repository. The measured evaluation artifacts are packaged with the release and can be reproduced using the frozen split and seed 42.
