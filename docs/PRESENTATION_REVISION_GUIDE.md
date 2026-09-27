# Presentation Revision Guide

## Problem statement

Use precise wording: a translation can be fluent and highly similar while losing a critical semantic feature such as negation, modality, quantity, condition, agency, or a protected term.

## Solution statement

Describe AMANAH as a semantic-integrity layer that compares a candidate translation with a trusted source and multiple trusted references, detects typed semantic drift, estimates severity and confidence, shows evidence, and abstains when verification is not reliable.

## Workflow slide

Use:

```text
Source verification
        ↓
Custom semantic analysis
        ↓
Decision fusion
        ↓
Evidence + human review
        ↓
Optional explanation / correction assistance
```

The LLM should be presented as an explanation and review-assistance component, not the final semantic judge.

## Technology slide

Add:

- multilingual fine-tuned drift classifier;
- AMANAH-SD controlled semantic-mutation pipeline;
- Tanzil canonical Arabic;
- three QuranEnc English references with provenance/version tracking;
- Hugging Face custom model endpoint;
- Cloudflare LLM explanation layer.

## Reliability slide

Include:

- immutable canonical/reference data;
- multi-reference comparison;
- PASS / REVIEW / CRITICAL / ABSTAIN;
- fail-closed behavior;
- human review for sensitive and uncertain cases.

## Metrics slide

Add metrics only after the frozen held-out evaluation completes:

- Critical Drift Recall;
- False Safe Rate;
- Macro F1;
- Severity Macro F1;
- per-label F1.

Remove any unmeasured percentage such as review-time reduction.

## Impact wording

Prefer: "reduce the risk of unintended semantic change during translation" rather than absolute guarantees.

## Team slide

Use role titles only in technical documentation and shared files, for example:

- ML, Dataset & Evaluation Lead
- Platform & Integration Lead
- Product / UX & Review Lead

No personal names are required in technical deliverables.
