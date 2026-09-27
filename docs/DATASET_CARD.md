# Dataset Card — AMANAH-SD v0

## Scope

Qur'an only, Arabic → English, designed around 6,236 ayah groups when the complete licensed source bundle is supplied.

## Immutable data classes

- `CANONICAL`: Arabic Qur'anic source; never modified.
- `REFERENCE`: trusted published English translation; never modified.
- `CANDIDATE`: trusted copy, human-authored candidate, or explicitly labeled synthetic mutation.

## Reference provenance

Every reference must store provider, translator/publisher, version, HTTPS source URL, retrieval timestamp, and SHA-256. The loader fails closed when provenance is incomplete.

## Synthetic mutations

Current deterministic operators:

- negation flip;
- quantifier change;
- modality shift;
- omission of a clearly coordinated trailing clause;
- condition loss.

Each operator is eligibility-gated. `validate_mutation` rejects unchanged, empty, provenance-inconsistent, label-inconsistent, and obvious multi-operation corruption.

## Split policy

The split unit is `ayah_id`, never individual rows. Every faithful/reference/mutated sample derived from one ayah stays in exactly one of train/validation/test. An optional unseen-surah holdout can be constructed separately.

## Gold benchmark

Synthetic data is suitable for supervised training and adversarial tests, but the final judging benchmark should include a human-reviewed subset. S3 examples should ideally be reviewed by both a language/translation reviewer and an Islamic-content specialist.

## Licensing and redistribution

The repository does not bundle a full third-party corpus. Download source material only under its provider terms and preserve version/source attribution. Do not publish transformed third-party data until redistribution rights for the derived dataset have been reviewed.
