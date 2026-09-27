---
library_name: transformers
pipeline_tag: text-classification
tags:
  - endpoints-template
  - semantic-integrity
  - multilingual
  - quran
---

# Model Card — AMANAH Drift Classifier v0.1

## Model purpose

Detect typed semantic drift between a trusted Arabic Qur'anic source and a candidate English translation. The classifier is one component of a hybrid system and is not a standalone religious authority.

## Base architecture

Recommended base: multilingual NLI-oriented mDeBERTa-v3-base. AMANAH adds:

- multi-label semantic-drift head;
- four-class severity head (`S0`–`S3`);
- per-label validation thresholds;
- deterministic safety rules and decision fusion outside the neural model.

## Labels

**Active trainable labels in v0.1:** `FAITHFUL`, `NEGATION_FLIP`, `OMISSION`, `MODALITY_SHIFT`, `QUANTIFIER_CHANGE`, `CONDITION_LOSS`. These are the classes currently generated/validated by the deterministic AMANAH-SD v0 pipeline.

**Reserved taxonomy for later validated datasets:** `ADDITION`, `ENTITY_SWAP`, `AGENCY_SHIFT`, `TEMPORAL_SHIFT`, `TERM_FLATTENING`, `LEXICAL_SEMANTIC_SHIFT`, `SEMANTIC_NARROWING`, `SEMANTIC_BROADENING`, `SEMANTIC_GRADATION_LOSS`, `INTERPRETATION_ADDITION`, `UNCERTAIN`. They remain in the domain schema but must not be advertised as trained model outputs until labeled training/validation data exists for them.

## Intended use

- pre-publication translation quality assurance;
- prioritizing passages for human review;
- benchmarking critical semantic drift;
- institutional integration through the AMANAH API.

## Not intended for

- issuing fatwas;
- replacing a qualified translator/reviewer;
- claiming a human translation is infallible;
- changing canonical Qur'anic text;
- evaluating languages or domains not represented in the measured benchmark without separate validation.

## Safety behavior

- model unavailable → `ABSTAIN`;
- reference missing → `ABSTAIN`;
- low confidence → `ABSTAIN`;
- high-precision S3 rule → `CRITICAL`;
- non-critical disagreement/findings → `REVIEW`;
- critical/uncertain cases → human review.

## Metrics

Populate this table **only after GPU training + frozen held-out evaluation**.

| Metric | Measured value |
|---|---:|
| Critical Drift Recall | Not measured yet |
| False Safe Rate | Not measured yet |
| Macro F1 | Not measured yet |
| Severity Macro F1 | Not measured yet |
| Unseen-surah performance | Not measured yet |

## Known limitations

- Synthetic mutations can differ from naturally occurring translation errors.
- Lexical rules are deliberately conservative and English-oriented.
- Span evidence in v0 is rule/alignment based rather than a trained token-classification head.
- Multiple trusted translations may disagree; those cases require explicit review rather than majority-vote truth.
- The current repository contains the training/deployment pipeline, not a claimed production checkpoint.
