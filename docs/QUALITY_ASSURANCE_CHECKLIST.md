# Quality Assurance Checklist

## Code

- [ ] Unit and contract tests pass.
- [ ] Training architecture smoke passes.
- [ ] Python modules compile successfully.
- [ ] No secrets or tokens are committed.
- [ ] Generated corpora and checkpoints remain ignored.

## Data

- [ ] Tanzil retrieval returns 6,236 canonical ayat.
- [ ] All three approved QuranEnc references are available.
- [ ] Every source/reference record contains provenance metadata.
- [ ] Canonical Arabic remains unchanged.
- [ ] Train/validation/test contain no shared ayah IDs.
- [ ] All active training labels are represented.

## Training

- [ ] GPU runtime is active.
- [ ] Best checkpoint is selected using validation loss.
- [ ] Thresholds are calibrated on validation only.
- [ ] Frozen test evaluation runs after model/threshold selection.
- [ ] Only measured metrics are copied to presentation material.

## Deployment

- [ ] Hugging Face package contains handler, checkpoint, thresholds, tokenizer files, reference store, and requirements.
- [ ] Local handler smoke test passes.
- [ ] Production endpoint returns the documented response schema.
- [ ] Platform integration keeps model results authoritative.
- [ ] Endpoint failure returns ABSTAIN and human review.
