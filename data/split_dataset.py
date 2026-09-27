from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from data.models import AmanahSample


def group_split(samples: list[AmanahSample], train_ratio: float, validation_ratio: float, test_ratio: float, seed: int = 42) -> dict[str, list[AmanahSample]]:
    total = train_ratio + validation_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1.0")
    groups: dict[str, list[AmanahSample]] = defaultdict(list)
    for sample in samples:
        groups[sample.ayah_id].append(sample)
    ids = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_ratio)
    n_val = int(n * validation_ratio)
    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])
    return {
        "train": [s for i in ids if i in train_ids for s in groups[i]],
        "validation": [s for i in ids if i in val_ids for s in groups[i]],
        "test": [s for i in ids if i in test_ids for s in groups[i]],
    }


def assert_no_ayah_leakage(splits: dict[str, list[AmanahSample]]) -> None:
    seen: dict[str, str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            previous = seen.setdefault(row.ayah_id, split_name)
            if previous != split_name:
                raise AssertionError(f"ayah leakage: {row.ayah_id} appears in {previous} and {split_name}")


def make_unseen_surah_holdout(samples: list[AmanahSample], holdout_surahs: set[int]) -> tuple[list[AmanahSample], list[AmanahSample]]:
    base = [s for s in samples if s.surah not in holdout_surahs]
    holdout = [s for s in samples if s.surah in holdout_surahs]
    return base, holdout
