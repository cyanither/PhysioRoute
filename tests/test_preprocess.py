from __future__ import annotations

import numpy as np

from src.data.preprocess import iter_window_indices, resample_labels_nearest, stress_onset_target
from src.data.splits import subject_disjoint_split


def test_iter_window_indices() -> None:
    starts = list(iter_window_indices(length=100, input_steps=20, horizon_steps=10, stride_steps=15))
    assert starts == [0, 15, 30, 45, 60]


def test_resample_labels_nearest() -> None:
    labels = np.array([0, 1, 2, 3], dtype=np.int16)
    out = resample_labels_nearest(labels, source_fs=1, target_fs=2, target_len=8)
    assert out.tolist() == [0, 0, 1, 2, 2, 2, 3, 3]


def test_stress_onset_target() -> None:
    current = np.array([1, 1, 1])
    future = np.array([1, 2, 2])
    assert stress_onset_target(current, future, stress_label=2, onset_only=True) == 1.0
    assert stress_onset_target(np.array([2]), future, stress_label=2, onset_only=True) == 0.0


def test_subject_disjoint_split() -> None:
    split = subject_disjoint_split([f"S{i}" for i in range(1, 16)], 0.7, 0.15, 0.15, seed=42)
    assert set(split.train).isdisjoint(split.val)
    assert set(split.train).isdisjoint(split.test)
    assert set(split.val).isdisjoint(split.test)
    assert len(split.train) + len(split.val) + len(split.test) == 15
