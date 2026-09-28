# Project Baseline

**Snapshot date:** 27 September 2026

This document records the implementation baseline before the measured GPU training run.

## Implemented

- Semantic-integrity workflow for Qur'an Arabic → English
- Data engineering pipeline with immutable canonical/reference/candidate separation
- Controlled semantic mutation operators for the active v0 label set
- Leakage-safe ayah-group train/validation/test splitting
- Multilingual classifier training pipeline
- Validation-based checkpoint selection
- Class weighting and threshold calibration
- Frozen-test evaluation pipeline
- Deterministic semantic safety rules and decision fusion
- PASS / REVIEW / CRITICAL / ABSTAIN response contract
- FastAPI deployment path
- Hugging Face custom inference handler
- Cloudflare integration adapter
- Automated unit and contract QA
- GitHub Actions CI

## Pending measured outputs

These are not claimed as complete until the GPU run finishes:

- Final model accuracy/F1/critical recall
- Production Hugging Face model repository
- Production inference endpoint
- Human-reviewed gold benchmark
- Measured review-time reduction

Any presentation or technical report should use only measured values from the final validation report.
