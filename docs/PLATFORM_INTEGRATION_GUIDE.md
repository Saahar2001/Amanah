# Platform Integration Guide

## Objective

Keep the existing web application and replace the current single-LLM decision path with a structured AMANAH model call.

```text
Arabic source + candidate English
        ↓
AMANAH model endpoint
        ↓
structured decision + evidence
        ↓
LLM explanation / wording assistance only
        ↓
result UI + human review
```

## Required server-side secrets

```text
AMANAH_ML_URL
AMANAH_ML_TOKEN
```

Do not expose either value to the browser.

## Integration helper

Use `integration-cloudflare.ts`. For a Hugging Face custom endpoint, call `analyzeAndExplain` with `transport: "hf"`.

The endpoint response fields `decision`, `severity`, `drifts`, `integrity_score`, and `reference_status` are authoritative. The explanation layer must not overwrite them.

## UI mapping

| Decision | UI state | Recommended wording |
|---|---|---|
| PASS | green | No material semantic drift detected by this analysis |
| REVIEW | amber | Potential semantic drift — human review recommended |
| CRITICAL | red | High-impact semantic drift detected — review required |
| ABSTAIN | gray | Unable to verify safely — human review required |

Do not describe PASS as religious certification or guaranteed correctness.

## Scope

The measured v0 classifier covers Qur'an Arabic → English. Any Hadith option should either be hidden for the measured demo or explicitly marked experimental and outside the trained benchmark.

## Failure behavior

If the model endpoint is unavailable, return `ABSTAIN`. Do not fall back to an unstructured LLM verdict.
