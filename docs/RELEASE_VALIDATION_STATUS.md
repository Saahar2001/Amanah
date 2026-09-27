# Release Validation Status

## Current source-code validation

The repository includes deterministic unit, contract, API, dataset, training, deployment, and fail-closed behavior tests.

## CI design

The main CI workflow runs local deterministic checks only. Live third-party source validation is intentionally outside the blocking CI workflow and is executed from the training notebook before dataset construction.

This prevents temporary external service/network issues from creating false build failures.

## Pending external validation

The following are complete only after execution in the relevant environment:

1. full 6,236-ayah source retrieval and three-reference verification;
2. GPU fine-tuning;
3. validation-only threshold calibration;
4. frozen held-out test evaluation;
5. measured checkpoint upload;
6. production inference endpoint health check;
7. application-to-endpoint integration smoke test.

No accuracy or F1 value should be claimed before the measured evaluation artifact is produced.
