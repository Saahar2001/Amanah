# AMANAH Evaluation Protocol

## Primary objective

Minimize **false safe** outcomes on critical semantic drift. The primary KPI is Critical Drift Recall.

## Required metrics

- Critical Drift Recall
- False Safe Rate
- Macro F1
- Per-label F1
- Severity Macro F1
- Optional calibration error and reviewer agreement

## Benchmark layers

1. Trusted/faithful candidate set
2. Controlled AMANAH-SD mutations
3. Minimal Semantic Perturbation Test (`examples/minimal_perturbations.json`)
4. Human-reviewed gold subset
5. Unseen-surah holdout

## Baselines

- Lexical/token similarity for offline sanity checks
- Multilingual embedding cosine when `sentence-transformers` is available
- Generic multilingual NLI when available
- Generic LLM judge as a comparator
- AMANAH custom classifier + rule fusion

Unavailable dependencies must be reported as `unavailable`; never replace a missing baseline with an invented score.

## Demo protocol

Use a minimal perturbation such as removal of `not`, `some → all`, or `may → must`. Show that lexical similarity can remain high while the protected semantic feature changes, then show AMANAH's typed finding and severity.
