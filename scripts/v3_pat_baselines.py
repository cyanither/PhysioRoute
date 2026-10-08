#!/usr/bin/env python3
"""Input-only signal-processing baselines for future PAT on v3 PTT-PPG."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal


ROOT = Path("data/processed_v3/ptt_ppg_h8")


def ecg_peaks(ecg: np.ndarray, fs: float = 64.0) -> np.ndarray:
    """Detect signed ECG peaks from one normalized input window."""

    centered = np.asarray(ecg, dtype=np.float64).reshape(-1) - np.median(ecg)
    scale = np.std(centered)
    if not np.isfinite(scale) or scale < 1e-6:
        return np.empty(0, dtype=np.int64)
    sign = 1.0 if np.quantile(centered, 0.995) >= -np.quantile(centered, 0.005) else -1.0
    peaks, _ = signal.find_peaks(
        sign * centered,
        distance=int(0.3 * fs),
        prominence=0.6 * scale,
    )
    return peaks


def peak_matching(ecg: np.ndarray, ppg: np.ndarray, fs: float = 64.0) -> float:
    """Median input-window delay from an ECG peak to maximum PPG upstroke."""

    peaks = ecg_peaks(ecg, fs)
    derivative = np.gradient(np.asarray(ppg, dtype=np.float64).reshape(-1))
    delays: list[float] = []
    for peak in peaks:
        lo = int(peak) + int(round(0.08 * fs))
        hi = min(int(peak) + int(round(0.45 * fs)), derivative.size - 1)
        if lo < hi:
            upstroke = lo + int(np.argmax(derivative[lo : hi + 1]))
            delays.append((upstroke - int(peak)) / fs)
    return float(np.median(delays)) if delays else float("nan")


def restricted_cross_correlation(ecg: np.ndarray, ppg: np.ndarray, fs: float = 64.0) -> float:
    """Lag maximizing R-impulse-to-positive-PPG-derivative correlation."""

    peaks = ecg_peaks(ecg, fs)
    if peaks.size < 2:
        return float("nan")
    impulses = np.zeros(np.asarray(ecg).size, dtype=np.float64)
    impulses[peaks] = 1.0
    upstroke = np.maximum(np.gradient(np.asarray(ppg, dtype=np.float64).reshape(-1)), 0.0)
    lags = np.arange(int(round(0.08 * fs)), int(round(0.45 * fs)) + 1)
    scores = np.full(lags.shape, -np.inf, dtype=np.float64)
    for index, lag in enumerate(lags):
        left = impulses[:-lag] - np.mean(impulses[:-lag])
        right = upstroke[lag:] - np.mean(upstroke[lag:])
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        if denominator > 1e-12:
            scores[index] = float(np.dot(left, right) / denominator)
    return float(lags[int(np.argmax(scores))] / fs) if np.isfinite(scores).any() else float("nan")


def metrics(prediction: np.ndarray, target: np.ndarray, subject: np.ndarray) -> dict[str, object]:
    """Return window and subject-macro PAT MAE in milliseconds."""

    prediction = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    subject = np.asarray(subject).astype(str).reshape(-1)
    valid = np.isfinite(prediction) & np.isfinite(target)
    per_subject = {
        value: float(np.mean(np.abs(prediction[valid & (subject == value)] - target[valid & (subject == value)])) * 1000.0)
        for value in np.unique(subject)
        if np.any(valid & (subject == value))
    }
    return {
        "window_mae_ms": float(np.mean(np.abs(prediction[valid] - target[valid])) * 1000.0),
        "subject_macro_mae_ms": float(np.mean(list(per_subject.values()))),
        "per_subject_mae_ms": per_subject,
        "valid_predictions": int(valid.sum()),
    }


def main() -> None:
    metadata = json.loads((ROOT / "metadata.json").read_text(encoding="utf-8"))
    node_names = list(metadata["node_names"])
    ecg_index = node_names.index("ecg")
    ppg_index = node_names.index("finger_ppg_ir")
    x = np.load(ROOT / "test/X.npy", mmap_mode="r")
    target = np.load(ROOT / "test/y_pat.npy", mmap_mode="r").reshape(-1)
    subject = np.load(ROOT / "test/subject.npy", mmap_mode="r").astype(str)
    training = np.load(ROOT / "train/y_pat.npy", mmap_mode="r").reshape(-1)
    median = float(np.nanmedian(training))
    persistence = np.load(ROOT / "test/baseline_current_pat.npy", mmap_mode="r").astype(np.float64)
    predictions = {
        "training_median": np.full(target.shape, median),
        "current_pat_persistence": persistence,
        "preprocessed_peak_matching": np.asarray(
            [peak_matching(sample[ecg_index, :, 0], sample[ppg_index, :, 0]) for sample in x]
        ),
        "restricted_cross_correlation": np.asarray(
            [restricted_cross_correlation(sample[ecg_index, :, 0], sample[ppg_index, :, 0]) for sample in x]
        ),
    }
    for values in predictions.values():
        values[~np.isfinite(values)] = median
    results = {name: metrics(values, target, subject) for name, values in predictions.items()}
    output = Path("results_v3/pat_signal_baselines.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
