# Source Provenance and License Notes

## Canonical Qur'anic Arabic

**Tanzil Quran Text — Uthmani, version 1.1**  
Source: https://tanzil.net/download/

The canonical text is stored as immutable source data. Synthetic mutations are never applied to canonical Qur'anic Arabic.

## English reference translations

**QuranEnc.com**  
API documentation: https://quranenc.com/en/home/api

Current v0 reference keys:

- `english_rwwad` — Rowwad Translation Center
- `english_saheeh` — Noor International Center
- `english_hilali_khan` — Hilali & Khan

Published reference translations are stored unchanged. Synthetic mutations are written only to separate candidate records and are never represented as published QuranEnc translations.

The full third-party corpus is intentionally not committed to this public repository. The retrieval pipeline records source key, version, retrieval timestamp, source URL, and checksums.

## Base model

Training base: `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`.

The model card and license should be verified again at release time, with required attribution retained in the deployed model repository.

## Repository code

Public visibility does not imply a blanket open-source license. No additional code license is granted unless a LICENSE file is explicitly added.
