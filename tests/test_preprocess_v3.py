from __future__ import annotations

import numpy as np

from src.data.preprocess_v3 import causal_align_signal, causal_window_zscore, future_hr_from_peaks


def test_causal_alignment_is_prefix_invariant() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=2000)
    altered = x.copy()
    altered[1200:] += 100.0
    y1 = causal_align_signal(x, source_fs=100, target_fs=50, target_len=1000, band=(0.5, 8.0))
    y2 = causal_align_signal(altered, source_fs=100, target_fs=50, target_len=1000, band=(0.5, 8.0))
    assert np.array_equal(y1[:600], y2[:600])


def test_window_normalization_uses_only_given_window() -> None:
    window = np.vstack([np.arange(10), np.ones(10)])
    out = causal_window_zscore(window)
    assert np.isclose(out[0].mean(), 0.0, atol=1e-6)
    assert np.isclose(out[0].std(), 1.0, atol=1e-6)
    assert np.all(out[1] == 0.0)


def test_future_hr_uses_complete_future_intervals() -> None:
    peaks = np.arange(0.0, 20.0, 1.0)
    assert np.isclose(future_hr_from_peaks(peaks, 5.0, 13.0), 60.0)
    assert np.isnan(future_hr_from_peaks(peaks, 5.1, 5.9))
