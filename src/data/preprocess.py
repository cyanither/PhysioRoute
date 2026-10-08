"""Preprocessing pipeline for wearable physiological datasets.

The first implemented dataset is WESAD. The pipeline reads subject pickle files,
filters and resamples configured channels, normalizes per subject, creates
subject-disjoint sliding windows, and writes split-specific memmapped ``.npy``
arrays for training.
"""

from __future__ import annotations

import json
import math
import pickle
import re
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from tqdm import tqdm

from src.data.splits import SplitSubjects, natural_key, subject_disjoint_split

try:  # Hydra is optional for unit tests that import pure functions.
    import hydra
    from omegaconf import DictConfig, OmegaConf
except Exception:  # pragma: no cover - exercised only in minimal environments.
    hydra = None
    DictConfig = Any
    OmegaConf = None


FILTER_BANDS: dict[str, tuple[float, float] | None] = {
    "ppg": (0.5, 8.0),
    "ecg": (0.5, 40.0),
    "resp": (0.1, 0.5),
    "none": None,
}


@dataclass(frozen=True)
class WindowTarget:
    """Targets computed for one window."""

    wave: np.ndarray
    hr: float
    hrv: float
    stress: float


def iter_window_indices(length: int, input_steps: int, horizon_steps: int, stride_steps: int) -> Iterator[int]:
    """Yield valid sliding-window start indices.

    Args:
        length: Total number of samples in the aligned sequence.
        input_steps: Number of samples in the input window.
        horizon_steps: Number of samples in the forecast horizon.
        stride_steps: Step between consecutive windows.

    Yields:
        Start indices where both input and forecast horizon fit.
    """

    if min(length, input_steps, horizon_steps, stride_steps) <= 0:
        return
    last_start = length - input_steps - horizon_steps
    if last_start < 0:
        return
    yield from range(0, last_start + 1, stride_steps)


def bandpass_filter(x: np.ndarray, fs: float, band: tuple[float, float] | None, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter when ``band`` is provided."""

    if band is None:
        return x.astype(np.float32, copy=False)
    low, high = band
    nyquist = fs / 2.0
    high = min(high, nyquist * 0.95)
    low = max(low, 1e-4)
    if not 0 < low < high < nyquist:
        return x.astype(np.float32, copy=False)
    min_len = max(32, order * 6)
    if x.size < min_len:
        return x.astype(np.float32, copy=False)
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def resample_signal(
    x: np.ndarray,
    source_fs: float,
    target_fs: float,
    target_len: int | None = None,
) -> np.ndarray:
    """Resample a 1D signal with polyphase filtering and optional trim/pad."""

    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("Cannot resample an empty signal.")
    if math.isclose(source_fs, target_fs):
        y = x.copy()
    else:
        ratio = Fraction(float(target_fs) / float(source_fs)).limit_denominator(1000)
        y = signal.resample_poly(x, ratio.numerator, ratio.denominator).astype(np.float32)
    if target_len is None:
        return y
    if y.size >= target_len:
        return y[:target_len]
    pad_value = y[-1] if y.size else 0.0
    return np.pad(y, (0, target_len - y.size), mode="constant", constant_values=pad_value)


def resample_labels_nearest(labels: np.ndarray, source_fs: float, target_fs: float, target_len: int) -> np.ndarray:
    """Resample discrete labels using nearest-neighbor time indexing."""

    labels = np.asarray(labels).reshape(-1)
    times = np.arange(target_len, dtype=np.float64) / float(target_fs)
    source_idx = np.rint(times * float(source_fs)).astype(np.int64)
    source_idx = np.clip(source_idx, 0, labels.size - 1)
    return labels[source_idx].astype(np.int16)


def zscore_per_node(x: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score normalize an aligned array with shape ``(nodes, time)`` per node."""

    mean = np.nanmean(x, axis=1, keepdims=True)
    std = np.nanstd(x, axis=1, keepdims=True)
    std = np.where(std < eps, 1.0, std)
    normalized = (x - mean) / std
    return normalized.astype(np.float32), mean.squeeze(1).astype(np.float32), std.squeeze(1).astype(np.float32)


def estimate_hr_hrv_from_ecg(ecg: np.ndarray, fs: float) -> tuple[float, float]:
    """Estimate mean HR and RMSSD HRV from an ECG segment.

    The implementation is intentionally dependency-light for preprocessing. It
    uses SciPy peak detection with conservative distance and prominence settings;
    later PTT-specific milestones can replace this with NeuroKit-derived peaks.
    """

    ecg = np.asarray(ecg, dtype=np.float32).reshape(-1)
    if ecg.size < int(fs * 3):
        return np.nan, np.nan
    centered = ecg - np.nanmedian(ecg)
    scale = np.nanstd(centered)
    if not np.isfinite(scale) or scale < 1e-6:
        return np.nan, np.nan
    min_distance = max(1, int(0.3 * fs))
    prominence = max(0.2 * scale, 1e-3)
    peaks, _ = signal.find_peaks(centered, distance=min_distance, prominence=prominence)
    if peaks.size < 2:
        return np.nan, np.nan
    rr_sec = np.diff(peaks) / float(fs)
    rr_sec = rr_sec[(rr_sec >= 0.3) & (rr_sec <= 2.0)]
    if rr_sec.size == 0:
        return np.nan, np.nan
    hr = 60.0 / float(np.nanmean(rr_sec))
    if rr_sec.size < 2:
        return hr, np.nan
    rmssd_ms = float(np.sqrt(np.nanmean(np.diff(rr_sec * 1000.0) ** 2)))
    return hr, rmssd_ms


def stress_onset_target(
    current_labels: np.ndarray,
    future_labels: np.ndarray,
    stress_label: int,
    onset_only: bool,
) -> float:
    """Compute a binary future stress or stress-onset target."""

    future_stress = bool(np.any(future_labels == stress_label))
    if not onset_only:
        return float(future_stress)
    current_stress = bool(np.any(current_labels == stress_label))
    return float(future_stress and not current_stress)


def discover_wesad_pickles(raw_dir: Path, max_subjects: int | None = None) -> list[Path]:
    """Return sorted WESAD subject pickle paths."""

    paths = sorted(raw_dir.glob("S*/S*.pkl"), key=lambda p: natural_key(p.parent.name))
    if max_subjects is not None:
        paths = paths[:max_subjects]
    if not paths:
        raise FileNotFoundError(f"No WESAD subject pickle files found under {raw_dir}.")
    return paths


def _load_wesad_pickle(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            return pickle.load(handle, encoding="latin1")


def _extract_wesad_node(data: Mapping[str, Any], node_cfg: Mapping[str, Any]) -> np.ndarray:
    source = str(node_cfg["source"])
    key = str(node_cfg["key"])
    signal_block = data["signal"][source][key]
    values = np.asarray(signal_block, dtype=np.float32)
    if values.ndim == 2:
        axis = int(node_cfg.get("axis", 0))
        values = values[:, axis]
    return values.reshape(-1)


def load_aligned_wesad_subject(
    path: Path,
    nodes: Sequence[Mapping[str, Any]],
    target_fs: float,
    label_fs: float,
) -> tuple[np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]:
    """Load, filter, resample, and normalize one WESAD subject.

    Returns:
        ``(signals, labels, node_names, normalization)`` where ``signals`` has
        shape ``(nodes, time)`` and labels are aligned to the same time axis.
    """

    raw = _load_wesad_pickle(path)
    raw_labels = np.asarray(raw["label"]).reshape(-1)
    target_len = int(math.floor(raw_labels.size / float(label_fs) * float(target_fs)))
    if target_len <= 0:
        raise ValueError(f"Subject {path.parent.name} has no aligned samples.")

    aligned: list[np.ndarray] = []
    node_names: list[str] = []
    for node_cfg in nodes:
        x = _extract_wesad_node(raw, node_cfg)
        filter_name = str(node_cfg.get("filter", "none"))
        band = FILTER_BANDS.get(filter_name)
        source_fs = float(node_cfg["fs"])
        filtered = bandpass_filter(x, source_fs, band)
        resampled = resample_signal(filtered, source_fs, target_fs, target_len=target_len)
        aligned.append(resampled)
        node_names.append(str(node_cfg["name"]))

    stacked = np.stack(aligned, axis=0)
    normalized, mean, std = zscore_per_node(stacked)
    labels = resample_labels_nearest(raw_labels, label_fs, target_fs, target_len=target_len)
    stats = {"mean": mean, "std": std}
    return normalized, labels, node_names, stats


def compute_wesad_target(
    signals: np.ndarray,
    labels: np.ndarray,
    node_names: Sequence[str],
    start: int,
    input_steps: int,
    horizon_steps: int,
    short_steps: int,
    fs: float,
    stress_label: int,
    onset_only: bool,
) -> WindowTarget:
    """Compute task targets for one WESAD window."""

    end = start + input_steps
    horizon_end = end + horizon_steps
    future_short_end = min(end + short_steps, signals.shape[1])

    bvp_idx = node_names.index("wrist_bvp")
    ecg_idx = node_names.index("ecg")
    wave = signals[bvp_idx, end:future_short_end]
    if wave.size < short_steps:
        wave = np.pad(wave, (0, short_steps - wave.size), mode="edge")

    hr, hrv = estimate_hr_hrv_from_ecg(signals[ecg_idx, end:horizon_end], fs)
    stress = stress_onset_target(
        labels[start:end],
        labels[end:horizon_end],
        stress_label=stress_label,
        onset_only=onset_only,
    )
    return WindowTarget(wave=wave.astype(np.float32), hr=hr, hrv=hrv, stress=stress)


def _split_map(splits: SplitSubjects) -> dict[str, str]:
    return {
        **{subject: "train" for subject in splits.train},
        **{subject: "val" for subject in splits.val},
        **{subject: "test" for subject in splits.test},
    }


def _window_count(sequence_len: int, input_steps: int, horizon_steps: int, stride_steps: int) -> int:
    return sum(1 for _ in iter_window_indices(sequence_len, input_steps, horizon_steps, stride_steps))


def _open_split_arrays(
    processed_dir: Path,
    split_counts: Mapping[str, int],
    n_nodes: int,
    input_steps: int,
    short_steps: int,
    dtype: str,
) -> dict[str, dict[str, np.ndarray]]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split, count in split_counts.items():
        split_dir = processed_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        arrays[split] = {
            "X": np.lib.format.open_memmap(
                split_dir / "X.npy",
                mode="w+",
                dtype=dtype,
                shape=(count, n_nodes, input_steps, 1),
            ),
            "y_wave": np.lib.format.open_memmap(
                split_dir / "y_wave.npy",
                mode="w+",
                dtype=dtype,
                shape=(count, short_steps),
            ),
            "y_hr": np.lib.format.open_memmap(
                split_dir / "y_hr.npy",
                mode="w+",
                dtype=dtype,
                shape=(count, 1),
            ),
            "y_hrv": np.lib.format.open_memmap(
                split_dir / "y_hrv.npy",
                mode="w+",
                dtype=dtype,
                shape=(count, 1),
            ),
            "y_stress": np.lib.format.open_memmap(
                split_dir / "y_stress.npy",
                mode="w+",
                dtype=dtype,
                shape=(count, 1),
            ),
            "start_idx": np.lib.format.open_memmap(
                split_dir / "start_idx.npy",
                mode="w+",
                dtype=np.int64,
                shape=(count,),
            ),
            "subject": np.lib.format.open_memmap(
                split_dir / "subject.npy",
                mode="w+",
                dtype="U16",
                shape=(count,),
            ),
        }
    return arrays


def preprocess_wesad(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Preprocess WESAD according to a config mapping."""

    raw_dir = Path(str(cfg["raw_dir"])).expanduser()
    processed_dir = Path(str(cfg["processed_dir"])).expanduser()
    processed_dir.mkdir(parents=True, exist_ok=True)

    target_fs = float(cfg["target_fs"])
    input_steps = int(round(float(cfg["input_sec"]) * target_fs))
    horizon_steps = int(round(float(cfg["horizon_sec"]) * target_fs))
    short_steps = int(round(float(cfg["short_horizon_sec"]) * target_fs))
    stride_steps = int(round(float(cfg["stride_sec"]) * target_fs))
    label_fs = float(cfg.get("label_fs", 700))
    output_dtype = str(cfg.get("output_dtype", "float32"))
    max_subjects = cfg.get("max_subjects")
    max_subjects = None if max_subjects in (None, "null") else int(max_subjects)

    subject_paths = discover_wesad_pickles(raw_dir, max_subjects=max_subjects)
    subjects = [path.parent.name for path in subject_paths]
    split_cfg = cfg["subject_split"]
    splits = subject_disjoint_split(
        subjects,
        train_ratio=float(split_cfg["train"]),
        val_ratio=float(split_cfg["val"]),
        test_ratio=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
    )
    subject_to_split = _split_map(splits)

    split_counts = {"train": 0, "val": 0, "test": 0}
    for path in subject_paths:
        raw = _load_wesad_pickle(path)
        target_len = int(math.floor(len(raw["label"]) / label_fs * target_fs))
        count = _window_count(target_len, input_steps, horizon_steps, stride_steps)
        split_counts[subject_to_split[path.parent.name]] += count

    nodes = list(cfg["nodes"])
    node_names = [str(node["name"]) for node in nodes]
    arrays = _open_split_arrays(
        processed_dir=processed_dir,
        split_counts=split_counts,
        n_nodes=len(nodes),
        input_steps=input_steps,
        short_steps=short_steps,
        dtype=output_dtype,
    )
    write_pos = {"train": 0, "val": 0, "test": 0}
    subject_stats: dict[str, dict[str, list[float]]] = {}

    for path in tqdm(subject_paths, desc="Preprocessing WESAD subjects"):
        subject = path.parent.name
        split = subject_to_split[subject]
        signals, labels, loaded_node_names, stats = load_aligned_wesad_subject(
            path=path,
            nodes=nodes,
            target_fs=target_fs,
            label_fs=label_fs,
        )
        if loaded_node_names != node_names:
            raise RuntimeError(f"Node-name mismatch for {subject}: {loaded_node_names} != {node_names}")
        subject_stats[subject] = {
            "mean": stats["mean"].astype(float).tolist(),
            "std": stats["std"].astype(float).tolist(),
        }

        for start in iter_window_indices(signals.shape[1], input_steps, horizon_steps, stride_steps):
            idx = write_pos[split]
            target = compute_wesad_target(
                signals=signals,
                labels=labels,
                node_names=node_names,
                start=start,
                input_steps=input_steps,
                horizon_steps=horizon_steps,
                short_steps=short_steps,
                fs=target_fs,
                stress_label=int(cfg.get("stress_label", 2)),
                onset_only=bool(cfg.get("stress_onset_only", True)),
            )
            arrays[split]["X"][idx] = signals[:, start : start + input_steps, None]
            arrays[split]["y_wave"][idx] = target.wave
            arrays[split]["y_hr"][idx, 0] = target.hr
            arrays[split]["y_hrv"][idx, 0] = target.hrv
            arrays[split]["y_stress"][idx, 0] = target.stress
            arrays[split]["start_idx"][idx] = start
            arrays[split]["subject"][idx] = subject
            write_pos[split] += 1

    for split_arrays in arrays.values():
        for array in split_arrays.values():
            if hasattr(array, "flush"):
                array.flush()

    metadata = {
        "dataset": "wesad",
        "target_fs": target_fs,
        "input_steps": input_steps,
        "horizon_steps": horizon_steps,
        "short_steps": short_steps,
        "stride_steps": stride_steps,
        "node_names": node_names,
        "splits": {"train": splits.train, "val": splits.val, "test": splits.test},
        "split_counts": split_counts,
        "subject_stats": subject_stats,
        "targets": ["wave", "hr", "hrv", "stress"],
    }
    with (processed_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def discover_ppg_dalia_pickles(raw_dir: Path, max_subjects: int | None = None) -> list[Path]:
    """Return sorted readable PPG-DaLiA subject pickle paths."""

    candidates = sorted(raw_dir.glob("S*/S*.pkl"), key=lambda p: natural_key(p.parent.name))
    if max_subjects is not None:
        candidates = candidates[:max_subjects]
    paths: list[Path] = []
    skipped: list[str] = []
    for path in candidates:
        try:
            _load_wesad_pickle(path)
            paths.append(path)
        except Exception as exc:
            skipped.append(f"{path}: {exc}")
    if skipped:
        print("Skipping unreadable PPG-DaLiA pickle files:")
        for item in skipped:
            print(f"  {item}")
    if not paths:
        raise FileNotFoundError(f"No readable PPG-DaLiA subject pickle files found under {raw_dir}.")
    return paths


def load_aligned_ppg_dalia_subject(
    path: Path,
    nodes: Sequence[Mapping[str, Any]],
    target_fs: float,
    hr_label_fs: float,
    activity_fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, np.ndarray]]:
    """Load, filter, resample, and normalize one PPG-DaLiA subject."""

    raw = _load_wesad_pickle(path)
    bvp = np.asarray(raw["signal"]["wrist"]["BVP"]).reshape(-1)
    target_len = int(math.floor(bvp.size / 64.0 * float(target_fs)))
    if target_len <= 0:
        raise ValueError(f"Subject {path.parent.name} has no aligned samples.")

    aligned: list[np.ndarray] = []
    node_names: list[str] = []
    for node_cfg in nodes:
        x = _extract_wesad_node(raw, node_cfg)
        filter_name = str(node_cfg.get("filter", "none"))
        band = FILTER_BANDS.get(filter_name)
        source_fs = float(node_cfg["fs"])
        filtered = bandpass_filter(x, source_fs, band)
        resampled = resample_signal(filtered, source_fs, target_fs, target_len=target_len)
        aligned.append(resampled)
        node_names.append(str(node_cfg["name"]))

    stacked = np.stack(aligned, axis=0)
    normalized, mean, std = zscore_per_node(stacked)
    hr = resample_signal(np.asarray(raw["label"]).reshape(-1), hr_label_fs, target_fs, target_len)
    activity = resample_labels_nearest(
        np.asarray(raw["activity"]).reshape(-1), activity_fs, target_fs, target_len
    )
    stats = {"mean": mean, "std": std}
    return normalized, hr.astype(np.float32), activity.astype(np.int16), node_names, stats


def _mode_int(values: np.ndarray) -> int:
    """Return an integer mode for a small label vector."""

    values = np.asarray(values).reshape(-1).astype(np.int64)
    if values.size == 0:
        return -1
    labels, counts = np.unique(values, return_counts=True)
    return int(labels[np.argmax(counts)])


def compute_ppg_dalia_target(
    signals: np.ndarray,
    hr_labels: np.ndarray,
    activity_labels: np.ndarray,
    node_names: Sequence[str],
    start: int,
    input_steps: int,
    horizon_steps: int,
    short_steps: int,
) -> tuple[WindowTarget, int]:
    """Compute waveform, HR, and activity targets for one PPG-DaLiA window."""

    end = start + input_steps
    horizon_end = end + horizon_steps
    future_short_end = min(end + short_steps, signals.shape[1])
    bvp_idx = node_names.index("wrist_bvp")
    wave = signals[bvp_idx, end:future_short_end]
    if wave.size < short_steps:
        wave = np.pad(wave, (0, short_steps - wave.size), mode="edge")
    future_hr = hr_labels[end:horizon_end]
    hr = float(np.nanmean(future_hr)) if future_hr.size else np.nan
    activity = _mode_int(activity_labels[start:end])
    return WindowTarget(wave=wave.astype(np.float32), hr=hr, hrv=np.nan, stress=0.0), activity


def preprocess_ppg_dalia(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Preprocess PPG-DaLiA according to a config mapping."""

    raw_dir = Path(str(cfg["raw_dir"])).expanduser()
    processed_dir = Path(str(cfg["processed_dir"])).expanduser()
    processed_dir.mkdir(parents=True, exist_ok=True)

    target_fs = float(cfg["target_fs"])
    input_steps = int(round(float(cfg["input_sec"]) * target_fs))
    horizon_steps = int(round(float(cfg["horizon_sec"]) * target_fs))
    short_steps = int(round(float(cfg["short_horizon_sec"]) * target_fs))
    required_horizon_steps = max(horizon_steps, short_steps)
    stride_steps = int(round(float(cfg["stride_sec"]) * target_fs))
    hr_label_fs = float(cfg.get("hr_label_fs", 0.5))
    activity_fs = float(cfg.get("activity_fs", 4))
    output_dtype = str(cfg.get("output_dtype", "float32"))
    max_subjects = cfg.get("max_subjects")
    max_subjects = None if max_subjects in (None, "null") else int(max_subjects)

    subject_paths = discover_ppg_dalia_pickles(raw_dir, max_subjects=max_subjects)
    subjects = [path.parent.name for path in subject_paths]
    split_cfg = cfg["subject_split"]
    splits = subject_disjoint_split(
        subjects,
        train_ratio=float(split_cfg["train"]),
        val_ratio=float(split_cfg["val"]),
        test_ratio=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
    )
    subject_to_split = _split_map(splits)

    split_counts = {"train": 0, "val": 0, "test": 0}
    for path in subject_paths:
        raw = _load_wesad_pickle(path)
        bvp_len = np.asarray(raw["signal"]["wrist"]["BVP"]).reshape(-1).size
        target_len = int(math.floor(bvp_len / 64.0 * target_fs))
        count = _window_count(target_len, input_steps, required_horizon_steps, stride_steps)
        split_counts[subject_to_split[path.parent.name]] += count

    nodes = list(cfg["nodes"])
    node_names = [str(node["name"]) for node in nodes]
    arrays = _open_split_arrays(
        processed_dir=processed_dir,
        split_counts=split_counts,
        n_nodes=len(nodes),
        input_steps=input_steps,
        short_steps=short_steps,
        dtype=output_dtype,
    )
    for split, split_arrays in arrays.items():
        split_arrays["y_activity"] = np.lib.format.open_memmap(
            processed_dir / split / "y_activity.npy",
            mode="w+",
            dtype=np.int16,
            shape=(split_counts[split], 1),
        )

    write_pos = {"train": 0, "val": 0, "test": 0}
    subject_stats: dict[str, dict[str, list[float]]] = {}

    for path in tqdm(subject_paths, desc="Preprocessing PPG-DaLiA subjects"):
        subject = path.parent.name
        split = subject_to_split[subject]
        signals, hr_labels, activity_labels, loaded_node_names, stats = load_aligned_ppg_dalia_subject(
            path=path,
            nodes=nodes,
            target_fs=target_fs,
            hr_label_fs=hr_label_fs,
            activity_fs=activity_fs,
        )
        if loaded_node_names != node_names:
            raise RuntimeError(f"Node-name mismatch for {subject}: {loaded_node_names} != {node_names}")
        subject_stats[subject] = {
            "mean": stats["mean"].astype(float).tolist(),
            "std": stats["std"].astype(float).tolist(),
        }

        for start in iter_window_indices(
            signals.shape[1], input_steps, required_horizon_steps, stride_steps
        ):
            idx = write_pos[split]
            target, activity = compute_ppg_dalia_target(
                signals=signals,
                hr_labels=hr_labels,
                activity_labels=activity_labels,
                node_names=node_names,
                start=start,
                input_steps=input_steps,
                horizon_steps=horizon_steps,
                short_steps=short_steps,
            )
            arrays[split]["X"][idx] = signals[:, start : start + input_steps, None]
            arrays[split]["y_wave"][idx] = target.wave
            arrays[split]["y_hr"][idx, 0] = target.hr
            arrays[split]["y_hrv"][idx, 0] = target.hrv
            arrays[split]["y_stress"][idx, 0] = target.stress
            arrays[split]["y_activity"][idx, 0] = activity
            arrays[split]["start_idx"][idx] = start
            arrays[split]["subject"][idx] = subject
            write_pos[split] += 1

    for split_arrays in arrays.values():
        for array in split_arrays.values():
            if hasattr(array, "flush"):
                array.flush()

    metadata = {
        "dataset": "ppg_dalia",
        "target_fs": target_fs,
        "input_steps": input_steps,
        "horizon_steps": horizon_steps,
        "short_steps": short_steps,
        "stride_steps": stride_steps,
        "node_names": node_names,
        "splits": {"train": splits.train, "val": splits.val, "test": splits.test},
        "split_counts": split_counts,
        "subject_stats": subject_stats,
        "targets": ["wave", "hr", "hrv", "stress", "activity"],
    }
    with (processed_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def _resolve_ptt_root(raw_dir: Path) -> Path:
    """Resolve the directory containing PTT-PPG WFDB RECORDS."""

    candidates = [
        raw_dir,
        raw_dir / "physionet.org" / "files" / "pulse-transit-time-ppg" / "1.1.0",
    ]
    for candidate in candidates:
        if (candidate / "RECORDS").exists():
            return candidate
    raise FileNotFoundError(f"Could not find PTT-PPG RECORDS under {raw_dir}.")


def discover_ptt_records(raw_dir: Path, max_records: int | None = None) -> list[Path]:
    """Return sorted PTT-PPG WFDB record base paths."""

    root = _resolve_ptt_root(raw_dir)
    records = [line.strip() for line in (root / "RECORDS").read_text().splitlines() if line.strip()]
    paths = sorted((root / record for record in records), key=lambda p: natural_key(p.name))
    if max_records is not None:
        paths = paths[:max_records]
    if not paths:
        raise FileNotFoundError(f"No PTT-PPG records found in {root / 'RECORDS'}.")
    return paths


def _parse_ptt_subject(record_name: str) -> str:
    """Extract a subject ID such as ``S1`` from a PTT-PPG record name."""

    subject = record_name.split("_", maxsplit=1)[0]
    return subject.upper()


def _read_ptt_header_info(record_base: Path) -> tuple[float, int, dict[str, str]]:
    """Read WFDB header sampling metadata and participant numerics."""

    header = (record_base.with_suffix(".hea")).read_text().splitlines()
    first = header[0].split()
    fs = float(first[2])
    sig_len = int(first[3])
    metadata: dict[str, str] = {}
    for line in header:
        if line.startswith("#"):
            for key, value in re.findall(r"<([^>]+)>:\s*([^<]+)", line):
                metadata[key.strip()] = value.strip()
    return fs, sig_len, metadata


def load_aligned_ptt_record(
    record_base: Path,
    nodes: Sequence[Mapping[str, Any]],
    target_fs: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, list[str], dict[str, np.ndarray]]:
    """Load, filter, resample, and normalize one PTT-PPG WFDB record.

    Returns:
        Signals, BP trajectory, R-peak indices on the target grid, metadata vector
        ``[activity_code]``, source sampling rate, node names, and normalization stats.
    """

    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover - depends on optional runtime dependency.
        raise ImportError("PTT-PPG preprocessing requires the 'wfdb' package from requirements.txt.") from exc

    record = wfdb.rdrecord(str(record_base))
    source_fs = float(record.fs)
    signal_names = list(record.sig_name)
    raw_signals = np.asarray(record.p_signal, dtype=np.float32)
    if raw_signals.ndim != 2:
        raise ValueError(f"Expected 2D WFDB signal matrix for {record_base.name}.")
    target_len = int(math.floor(raw_signals.shape[0] / source_fs * float(target_fs)))

    aligned: list[np.ndarray] = []
    node_names: list[str] = []
    for node_cfg in nodes:
        key = str(node_cfg["key"])
        if key not in signal_names:
            raise KeyError(f"Channel {key!r} not found in {record_base.name}; available: {signal_names}")
        x = raw_signals[:, signal_names.index(key)]
        filter_name = str(node_cfg.get("filter", "none"))
        band = FILTER_BANDS.get(filter_name)
        filtered = bandpass_filter(x, source_fs, band)
        resampled = resample_signal(filtered, source_fs, target_fs, target_len=target_len)
        aligned.append(resampled)
        node_names.append(str(node_cfg["name"]))

    stacked = np.stack(aligned, axis=0)
    normalized, mean, std = zscore_per_node(stacked)
    _, _, metadata = _read_ptt_header_info(record_base)
    sbp_start = float(metadata.get("bp_sys_start", "nan"))
    sbp_end = float(metadata.get("bp_sys_end", "nan"))
    dbp_start = float(metadata.get("bp_dia_start", "nan"))
    dbp_end = float(metadata.get("bp_dia_end", "nan"))
    bp = np.stack(
        [
            np.linspace(sbp_start, sbp_end, target_len, dtype=np.float32),
            np.linspace(dbp_start, dbp_end, target_len, dtype=np.float32),
        ],
        axis=1,
    )
    activity = str(metadata.get("activity", "unknown"))
    activity_code = {"sit": 0, "walk": 1, "run": 2}.get(activity, -1)

    try:
        ann = wfdb.rdann(str(record_base), "atr")
        rpeaks = np.rint(np.asarray(ann.sample, dtype=np.float64) / source_fs * target_fs).astype(np.int64)
        rpeaks = rpeaks[(rpeaks >= 0) & (rpeaks < target_len)]
    except Exception:
        rpeaks = np.empty((0,), dtype=np.int64)

    stats = {"mean": mean, "std": std}
    return normalized, bp, rpeaks, np.asarray([activity_code], dtype=np.int16), int(source_fs), node_names, stats


def _hr_from_rpeaks(rpeaks: np.ndarray, start: int, end: int, fs: float) -> float:
    """Estimate HR from target-grid R peaks inside a window."""

    peaks = rpeaks[(rpeaks >= start) & (rpeaks < end)]
    if peaks.size < 2:
        return np.nan
    rr_sec = np.diff(peaks) / float(fs)
    rr_sec = rr_sec[(rr_sec >= 0.3) & (rr_sec <= 2.0)]
    if rr_sec.size == 0:
        return np.nan
    return 60.0 / float(np.nanmean(rr_sec))


def _ptt_from_rpeaks(
    signals: np.ndarray,
    node_names: Sequence[str],
    rpeaks: np.ndarray,
    start: int,
    end: int,
    fs: float,
    ppg_node: str = "finger_ppg_ir",
) -> float:
    """Estimate PTT as ECG R peak to following PPG peak delay in seconds."""

    if ppg_node not in node_names:
        return np.nan
    ppg = signals[node_names.index(ppg_node)]
    delays: list[float] = []
    for peak in rpeaks[(rpeaks >= start) & (rpeaks < end)]:
        lo = int(peak + round(0.10 * fs))
        hi = int(peak + round(0.40 * fs))
        if hi >= ppg.size or lo >= hi:
            continue
        local = ppg[lo:hi]
        if local.size == 0 or not np.isfinite(local).any():
            continue
        ppg_peak = lo + int(np.nanargmax(local))
        delays.append((ppg_peak - int(peak)) / float(fs))
    return float(np.nanmean(delays)) if delays else np.nan


def compute_ptt_target(
    signals: np.ndarray,
    bp: np.ndarray,
    rpeaks: np.ndarray,
    activity_code: int,
    node_names: Sequence[str],
    start: int,
    input_steps: int,
    horizon_steps: int,
    short_steps: int,
    fs: float,
) -> tuple[WindowTarget, np.ndarray, float, int]:
    """Compute waveform, HR, BP, PTT, and activity targets for one PTT-PPG window."""

    end = start + input_steps
    horizon_end = end + horizon_steps
    bvp_idx = node_names.index("finger_ppg_ir") if "finger_ppg_ir" in node_names else 1
    wave = signals[bvp_idx, end : end + short_steps]
    if wave.size < short_steps:
        wave = np.pad(wave, (0, short_steps - wave.size), mode="edge")
    bp_future = bp[end:horizon_end]
    if bp_future.shape[0] < horizon_steps:
        pad = np.repeat(bp_future[-1:, :], horizon_steps - bp_future.shape[0], axis=0)
        bp_future = np.concatenate([bp_future, pad], axis=0)
    hr = _hr_from_rpeaks(rpeaks, end, horizon_end, fs)
    ptt = _ptt_from_rpeaks(signals, node_names, rpeaks, start, horizon_end, fs)
    return WindowTarget(wave=wave.astype(np.float32), hr=hr, hrv=np.nan, stress=0.0), bp_future, ptt, activity_code


def preprocess_ptt_ppg(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Preprocess the WFDB PTT-PPG dataset according to a config mapping."""

    raw_dir = Path(str(cfg["raw_dir"])).expanduser()
    processed_dir = Path(str(cfg["processed_dir"])).expanduser()
    processed_dir.mkdir(parents=True, exist_ok=True)

    target_fs = float(cfg["target_fs"])
    input_steps = int(round(float(cfg["input_sec"]) * target_fs))
    horizon_steps = int(round(float(cfg["horizon_sec"]) * target_fs))
    short_steps = int(round(float(cfg["short_horizon_sec"]) * target_fs))
    stride_steps = int(round(float(cfg["stride_sec"]) * target_fs))
    output_dtype = str(cfg.get("output_dtype", "float32"))
    max_records = cfg.get("max_records")
    max_records = None if max_records in (None, "null") else int(max_records)

    record_paths = discover_ptt_records(raw_dir, max_records=max_records)
    subjects = sorted({_parse_ptt_subject(path.name) for path in record_paths}, key=natural_key)
    split_cfg = cfg["subject_split"]
    splits = subject_disjoint_split(
        subjects,
        train_ratio=float(split_cfg["train"]),
        val_ratio=float(split_cfg["val"]),
        test_ratio=float(split_cfg["test"]),
        seed=int(split_cfg["seed"]),
    )
    subject_to_split = _split_map(splits)

    split_counts = {"train": 0, "val": 0, "test": 0}
    for record_base in record_paths:
        source_fs, sig_len, _ = _read_ptt_header_info(record_base)
        target_len = int(math.floor(sig_len / source_fs * target_fs))
        split_counts[subject_to_split[_parse_ptt_subject(record_base.name)]] += _window_count(
            target_len, input_steps, horizon_steps, stride_steps
        )

    nodes = list(cfg["nodes"])
    node_names = [str(node["name"]) for node in nodes]
    arrays = _open_split_arrays(
        processed_dir=processed_dir,
        split_counts=split_counts,
        n_nodes=len(nodes),
        input_steps=input_steps,
        short_steps=short_steps,
        dtype=output_dtype,
    )
    for split, split_arrays in arrays.items():
        split_arrays["y_bp"] = np.lib.format.open_memmap(
            processed_dir / split / "y_bp.npy",
            mode="w+",
            dtype=output_dtype,
            shape=(split_counts[split], horizon_steps, 2),
        )
        split_arrays["y_ptt"] = np.lib.format.open_memmap(
            processed_dir / split / "y_ptt.npy",
            mode="w+",
            dtype=output_dtype,
            shape=(split_counts[split], 1),
        )
        split_arrays["y_activity"] = np.lib.format.open_memmap(
            processed_dir / split / "y_activity.npy",
            mode="w+",
            dtype=np.int16,
            shape=(split_counts[split], 1),
        )

    write_pos = {"train": 0, "val": 0, "test": 0}
    record_stats: dict[str, dict[str, list[float]]] = {}
    for record_base in tqdm(record_paths, desc="Preprocessing PTT-PPG records"):
        subject = _parse_ptt_subject(record_base.name)
        split = subject_to_split[subject]
        signals, bp, rpeaks, activity_vec, source_fs, loaded_node_names, stats = load_aligned_ptt_record(
            record_base=record_base,
            nodes=nodes,
            target_fs=target_fs,
        )
        if loaded_node_names != node_names:
            raise RuntimeError(f"Node-name mismatch for {record_base.name}: {loaded_node_names} != {node_names}")
        record_stats[record_base.name] = {
            "mean": stats["mean"].astype(float).tolist(),
            "std": stats["std"].astype(float).tolist(),
        }
        for start in iter_window_indices(signals.shape[1], input_steps, horizon_steps, stride_steps):
            idx = write_pos[split]
            target, bp_future, ptt, activity = compute_ptt_target(
                signals=signals,
                bp=bp,
                rpeaks=rpeaks,
                activity_code=int(activity_vec[0]),
                node_names=node_names,
                start=start,
                input_steps=input_steps,
                horizon_steps=horizon_steps,
                short_steps=short_steps,
                fs=target_fs,
            )
            arrays[split]["X"][idx] = signals[:, start : start + input_steps, None]
            arrays[split]["y_wave"][idx] = target.wave
            arrays[split]["y_hr"][idx, 0] = target.hr
            arrays[split]["y_hrv"][idx, 0] = target.hrv
            arrays[split]["y_stress"][idx, 0] = target.stress
            arrays[split]["y_bp"][idx] = bp_future
            arrays[split]["y_ptt"][idx, 0] = ptt
            arrays[split]["y_activity"][idx, 0] = activity
            arrays[split]["start_idx"][idx] = start
            arrays[split]["subject"][idx] = subject
            write_pos[split] += 1

    for split_arrays in arrays.values():
        for array in split_arrays.values():
            if hasattr(array, "flush"):
                array.flush()

    metadata = {
        "dataset": "ptt_ppg",
        "target_fs": target_fs,
        "input_steps": input_steps,
        "horizon_steps": horizon_steps,
        "short_steps": short_steps,
        "stride_steps": stride_steps,
        "node_names": node_names,
        "splits": {"train": splits.train, "val": splits.val, "test": splits.test},
        "split_counts": split_counts,
        "record_stats": record_stats,
        "targets": ["wave", "hr", "hrv", "stress", "bp", "ptt", "activity"],
        "activity_codes": {"sit": 0, "walk": 1, "run": 2},
    }
    with (processed_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def preprocess_dataset(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Dispatch preprocessing by dataset name."""

    name = str(cfg["name"]).lower()
    if name == "wesad":
        return preprocess_wesad(cfg)
    if name == "ppg_dalia":
        return preprocess_ppg_dalia(cfg)
    if name == "ptt_ppg":
        return preprocess_ptt_ppg(cfg)
    raise NotImplementedError(f"Preprocessing for dataset '{name}' is not implemented yet.")


def _to_plain_config(cfg: DictConfig | Mapping[str, Any]) -> Mapping[str, Any]:
    if OmegaConf is not None and isinstance(cfg, DictConfig):
        root = OmegaConf.to_container(cfg, resolve=True)
    else:
        root = cfg
    if not isinstance(root, Mapping):
        raise TypeError("Expected a mapping-like config.")
    data_cfg = root.get("data", root)
    if not isinstance(data_cfg, Mapping):
        raise TypeError("Expected config key 'data' to contain a mapping.")
    return data_cfg


if hydra is not None:

    @hydra.main(version_base=None, config_path="../../configs", config_name="preprocess")
    def main(cfg: DictConfig) -> None:
        """Hydra entry point."""

        metadata = preprocess_dataset(_to_plain_config(cfg))
        print(json.dumps({"processed": metadata["dataset"], "split_counts": metadata["split_counts"]}))

else:  # pragma: no cover

    def main(_: Any = None) -> None:
        raise RuntimeError("hydra-core is required for the command-line preprocessing entry point.")


if __name__ == "__main__":
    main()
