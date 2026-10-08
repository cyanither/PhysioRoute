"""Leakage-safe non-neural baselines for the v3 HR/PAT protocols."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOTS = {
    "WESAD": Path("data/processed_v3/wesad_h8"),
    "PPG-DaLiA": Path("data/processed_v3/ppg_dalia_h8"),
    "PTT-PPG": Path("data/processed_v3/ptt_ppg_h8"),
}


def _hr_from_ecg(ecg: np.ndarray, fs: float = 64.0) -> float:
    segment = np.asarray(ecg, dtype=np.float64).reshape(-1)
    centered = segment - np.median(segment)
    scale = np.std(centered)
    if not np.isfinite(scale) or scale < 1e-6:
        return float("nan")
    sign = 1.0 if np.quantile(centered, 0.995) >= -np.quantile(centered, 0.005) else -1.0
    peaks, _ = signal.find_peaks(sign * centered, distance=int(0.3 * fs), prominence=0.6 * scale)
    rr = np.diff(peaks) / fs
    rr = rr[(rr >= 0.3) & (rr <= 2.0)]
    return float(60.0 / np.mean(rr)) if rr.size else float("nan")


def _pat_from_input(ecg: np.ndarray, ppg: np.ndarray, fs: float = 64.0) -> float:
    ecg = np.asarray(ecg, dtype=np.float64)
    ppg = np.asarray(ppg, dtype=np.float64)
    centered = ecg - np.median(ecg)
    scale = np.std(centered)
    if scale < 1e-6:
        return float("nan")
    sign = 1.0 if np.quantile(centered, 0.995) >= -np.quantile(centered, 0.005) else -1.0
    peaks, _ = signal.find_peaks(sign * centered, distance=int(0.3 * fs), prominence=0.6 * scale)
    derivative = np.gradient(ppg)
    delays = []
    for peak in peaks:
        lo = peak + int(round(0.08 * fs))
        hi = min(peak + int(round(0.45 * fs)), ppg.size - 1)
        if lo < hi:
            upstroke = lo + int(np.argmax(derivative[lo : hi + 1]))
            delays.append((upstroke - peak) / fs)
    return float(np.median(delays)) if delays else float("nan")


def _summary_features(x: np.ndarray, fs: float = 64.0) -> np.ndarray:
    """Compact input-only waveform features for a linear ridge baseline."""

    tail = x[:, :, -int(8 * fs) :, 0]
    features = []
    for batch in tail:
        node_parts = []
        for channel in batch:
            diff = np.diff(channel)
            spectrum = np.abs(np.fft.rfft(channel)) ** 2
            freqs = np.fft.rfftfreq(channel.size, d=1.0 / fs)
            bands = []
            for low, high in ((0.1, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 16)):
                mask = (freqs >= low) & (freqs < high)
                bands.append(float(np.log1p(spectrum[mask].mean())) if mask.any() else 0.0)
            node_parts.extend(
                [float(channel.mean()), float(channel.std()), float(channel.min()), float(channel.max()),
                 float(channel[-1] - channel[0]), float(diff.std())] + bands
            )
        node_parts.append(_hr_from_ecg(batch[0], fs))
        features.append(node_parts)
    out = np.asarray(features, dtype=np.float32)
    for col in range(out.shape[1]):
        values = out[:, col]
        fill = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
        values[~np.isfinite(values)] = fill
    return out



def _ppg_acc_hr(
    x: np.ndarray, node_names: list[str], fs: float = 64.0, tail_sec: float = 30.0
) -> np.ndarray:
    """Estimate current HR from PPG PSD after suppressing accelerometer spectral peaks."""

    ppg_candidates = ("wrist_bvp", "finger_ppg_ir", "finger_ppg_red", "proximal_ppg_ir")
    ppg_idx = next((node_names.index(name) for name in ppg_candidates if name in node_names), None)
    if ppg_idx is None:
        return np.full(len(x), np.nan)
    acc_idx = [idx for idx, name in enumerate(node_names) if name.endswith(("acc_x", "acc_y", "acc_z"))]
    steps = int(round(tail_sec * fs))
    predictions = []
    for sample in x:
        ppg = np.asarray(sample[ppg_idx, -steps:, 0], dtype=np.float64)
        freq, ppg_psd = signal.welch(ppg, fs=fs, nperseg=min(512, len(ppg)))
        band = (freq >= 0.6) & (freq <= 3.5)
        if not band.any() or not np.isfinite(ppg_psd[band]).any():
            predictions.append(np.nan)
            continue
        score = ppg_psd[band] / max(float(np.max(ppg_psd[band])), 1e-12)
        if acc_idx:
            motion = np.zeros_like(ppg_psd)
            for idx in acc_idx:
                _, psd = signal.welch(
                    np.asarray(sample[idx, -steps:, 0], dtype=np.float64),
                    fs=fs,
                    nperseg=min(512, len(ppg)),
                )
                motion += psd
            motion_band = motion[band] / max(float(np.max(motion[band])), 1e-12)
            score = score - 0.5 * motion_band
        band_freq = freq[band]
        peak_hz = float(band_freq[int(np.argmax(score))])
        predictions.append(peak_hz * 60.0)
    return np.asarray(predictions, dtype=np.float64)


def _activity_mean_predictions(root: Path, split: str, fallback: float) -> np.ndarray:
    """Predict with training-only input-activity means; unknown codes use the global median."""

    train_activity = np.load(root / "train" / "activity.npy", mmap_mode="r").reshape(-1)
    train_hr = np.load(root / "train" / "y_hr.npy", mmap_mode="r").reshape(-1)
    means = {
        int(code): float(np.nanmean(train_hr[train_activity == code]))
        for code in np.unique(train_activity)
        if code >= 0 and np.any(train_activity == code)
    }
    activity = np.load(root / split / "activity.npy", mmap_mode="r").reshape(-1)
    return np.asarray([means.get(int(code), fallback) for code in activity], dtype=np.float64)

def _metrics(pred: np.ndarray, target: np.ndarray, subject: np.ndarray) -> dict[str, object]:
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    subject = np.asarray(subject).astype(str).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(target)
    per_subject = {
        value: float(np.mean(np.abs(pred[valid & (subject == value)] - target[valid & (subject == value)])))
        for value in np.unique(subject)
        if np.any(valid & (subject == value))
    }
    return {
        "window_mae": float(np.mean(np.abs(pred[valid] - target[valid]))),
        "subject_macro_mae": float(np.mean(list(per_subject.values()))),
        "per_subject_mae": per_subject,
        "valid_predictions": int(valid.sum()),
    }


def _read(root: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folder = root / split
    return (
        np.load(folder / "X.npy", mmap_mode="r"),
        np.load(folder / "y_hr.npy", mmap_mode="r").reshape(-1),
        np.load(folder / "subject.npy", mmap_mode="r").astype(str),
    )


def evaluate_dataset(root: Path) -> dict[str, object]:
    train_x, train_y, _ = _read(root, "train")
    test_x, test_y, test_subject = _read(root, "test")
    train_median = float(np.nanmedian(train_y))
    median_pred = np.full(test_y.shape, train_median)
    train_current_hr = np.load(root / "train" / "current_hr.npy", mmap_mode="r").astype(np.float64)
    test_current_hr = np.load(root / "test" / "current_hr.npy", mmap_mode="r").astype(np.float64)
    persistence = test_current_hr.copy()
    persistence[~np.isfinite(persistence)] = train_median

    metadata = json.loads((root / "metadata.json").read_text())
    node_names = list(metadata["node_names"])
    ppg_acc = []
    batch_size = 256
    for start in range(0, len(test_x), batch_size):
        ppg_acc.append(_ppg_acc_hr(test_x[start : start + batch_size], node_names))
    ppg_acc_pred = np.concatenate(ppg_acc)
    ppg_acc_pred[~np.isfinite(ppg_acc_pred)] = train_median

    train_features = np.concatenate(
        [_summary_features(train_x[start : start + batch_size]) for start in range(0, len(train_x), batch_size)]
    )
    test_features = np.concatenate(
        [_summary_features(test_x[start : start + batch_size]) for start in range(0, len(test_x), batch_size)]
    )
    valid_train = np.isfinite(train_y)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(train_features[valid_train], train_y[valid_train])
    ridge_pred = ridge.predict(test_features)

    ar_valid = np.isfinite(train_y) & np.isfinite(train_current_hr)
    ar_model = Ridge(alpha=1.0)
    ar_model.fit(train_current_hr[ar_valid, None], train_y[ar_valid] - train_current_hr[ar_valid])
    ar_pred = persistence + ar_model.predict(persistence[:, None])
    activity_pred = _activity_mean_predictions(root, "test", train_median)

    results: dict[str, object] = {
        "train_median_bpm": train_median,
        "train_median": _metrics(median_pred, test_y, test_subject),
        "current_hr_persistence": _metrics(persistence, test_y, test_subject),
        "linear_ar_current_hr": _metrics(ar_pred, test_y, test_subject),
        "activity_conditioned_train_mean": _metrics(activity_pred, test_y, test_subject),
        "ppg_acc_signal_processing": _metrics(ppg_acc_pred, test_y, test_subject),
        "linear_ridge_waveform_features": _metrics(ridge_pred, test_y, test_subject),
    }
    pat_path = root / "test" / "y_pat.npy"
    if pat_path.exists():
        train_pat = np.load(root / "train" / "y_pat.npy", mmap_mode="r").reshape(-1)
        test_pat = np.load(pat_path, mmap_mode="r").reshape(-1)
        pat_median = float(np.nanmedian(train_pat))
        pat_median_pred = np.full(test_pat.shape, pat_median)
        pat_persistence = np.load(root / "test" / "baseline_current_pat.npy", mmap_mode="r").astype(np.float64)
        pat_persistence[~np.isfinite(pat_persistence)] = pat_median
        results["pat_ms"] = {
            "train_median": _metrics(pat_median_pred * 1000.0, test_pat * 1000.0, test_subject),
            "current_pat_persistence": _metrics(pat_persistence * 1000.0, test_pat * 1000.0, test_subject),
        }
    return results


def main() -> None:
    results = {name: evaluate_dataset(root) for name, root in ROOTS.items()}
    output = Path("results_v3/simple_baselines.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
