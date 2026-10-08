"""Subject-disjoint dataset splitting utilities."""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitSubjects:
    """Subject IDs assigned to train, validation, and test splits."""

    train: list[str]
    val: list[str]
    test: list[str]


def natural_key(value: str) -> list[int | str]:
    """Return a key that sorts strings containing integers naturally."""

    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value)]


def subject_disjoint_split(
    subjects: Sequence[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> SplitSubjects:
    """Create reproducible subject-disjoint splits.

    Args:
        subjects: Subject identifiers.
        train_ratio: Fraction of subjects assigned to train.
        val_ratio: Fraction of subjects assigned to validation.
        test_ratio: Fraction of subjects assigned to test.
        seed: Random seed used for shuffling subjects.

    Returns:
        A ``SplitSubjects`` instance.

    Raises:
        ValueError: If ratios are invalid or fewer than three subjects are available.
    """

    if len(subjects) < 3:
        raise ValueError("At least three subjects are required for train/val/test splits.")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}.")

    ordered = sorted(set(subjects), key=natural_key)
    rng = random.Random(seed)
    rng.shuffle(ordered)

    n_total = len(ordered)
    n_train = max(1, int(round(n_total * train_ratio)))
    n_val = max(1, int(round(n_total * val_ratio)))
    if n_train + n_val >= n_total:
        n_train = max(1, n_total - 2)
        n_val = 1

    train = sorted(ordered[:n_train], key=natural_key)
    val = sorted(ordered[n_train : n_train + n_val], key=natural_key)
    test = sorted(ordered[n_train + n_val :], key=natural_key)
    return SplitSubjects(train=train, val=val, test=test)
