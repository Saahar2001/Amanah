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

## Quick QA

```bash
pytest -q
python -m training.train --smoke
python -m compileall -q amanah_engine api data training scripts deployment
git diff --check
```

See `docs/MAHMOUD_HANDOFF.md`, `docs/QA_CHECKLIST.md`, and `notebooks/AMANAH_Training_Colab.ipynb` for integration and training.
