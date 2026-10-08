"""Leakage-safe v3 preprocessing for the manuscript experiments.

The v3 protocol deliberately exposes only input-time information to the model:
signals are causally filtered/resampled and normalized from each input window.
The sole common prediction target is future ECG-derived mean heart rate.  The
PTT-PPG dataset additionally supplies a future ECG-R-to-PPG-upstroke pulse
arrival time (PAT) target; endpoint cuff pressures are never interpolated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal
from tqdm import tqdm

from src.data.preprocess import _extract_wesad_node, _load_wesad_pickle, _read_ptt_header_info
from src.data.preprocess import _resolve_ptt_root, discover_ptt_records, discover_wesad_pickles
from src.data.splits import SplitSubjects, natural_key, subject_disjoint_split


FILTER_BANDS: dict[str, tuple[float, float] | None] = {
    "ppg": (0.5, 8.0),
    "ecg": (0.5, 30.0),
    "resp": (0.1, 0.5),
    "none": None,
}


@dataclass
class V3Record:
    """One aligned record plus label-only event annotations."""

    subject: str
    record: str
    signals: np.ndarray
    rpeak_times: np.ndarray
    activity: np.ndarray | None = None
    pat_rpeak_times: np.ndarray | None = None
    pat_values: np.ndarray | None = None


def _forward_fill_finite(x: np.ndarray) -> np.ndarray:
    """Replace nonfinite values using only values available at or before a sample."""

    x = np.asarray(x, dtype=np.float64).reshape(-1).copy()
    finite = np.isfinite(x)
    if finite.all():
        return x
    if not finite.any():
        return np.zeros_like(x)
    first = int(np.flatnonzero(finite)[0])
    x[:first] = 0.0
    valid_idx = np.where(np.isfinite(x), np.arange(x.size), 0)
    np.maximum.accumulate(valid_idx, out=valid_idx)
    return x[valid_idx]


def _sos_causal(x: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Causal SOS filtering with a constant-input steady-state initialization."""

    x = _forward_fill_finite(x)
    zi = signal.sosfilt_zi(sos) * float(x[0])
    y, _ = signal.sosfilt(sos, x, zi=zi)
    return y


def causal_align_signal(
    x: np.ndarray,
    source_fs: float,
    target_fs: float,
    target_len: int,
    band: tuple[float, float] | None,
    order: int = 4,
) -> np.ndarray:
    """Causally anti-alias, align by previous-sample hold, and bandpass filter."""

    x = _forward_fill_finite(x)
    source_fs = float(source_fs)
    target_fs = float(target_fs)
    if source_fs > target_fs:
        cutoff = min(0.45 * target_fs, 0.45 * source_fs)
        sos_lp = signal.butter(order, cutoff, btype="lowpass", fs=source_fs, output="sos")
        x = _sos_causal(x, sos_lp)
    target_times = np.arange(target_len, dtype=np.float64) / target_fs
    source_idx = np.floor(target_times * source_fs + 1e-12).astype(np.int64)
    source_idx = np.clip(source_idx, 0, x.size - 1)
    aligned = x[source_idx]
    if band is not None:
        low, high = band
        high = min(float(high), target_fs * 0.475)
        if 0.0 < low < high:
            sos_bp = signal.butter(order, [low, high], btype="bandpass", fs=target_fs, output="sos")
            aligned = _sos_causal(aligned, sos_bp)
    return aligned.astype(np.float32)


def causal_window_zscore(window: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Normalize each channel using the current input window only."""

    mean = np.nanmean(window, axis=1, keepdims=True)
    std = np.nanstd(window, axis=1, keepdims=True)
    std = np.where(np.isfinite(std) & (std >= eps), std, 1.0)
    out = (window - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def previous_hold_labels(labels: np.ndarray, source_fs: float, target_fs: float, target_len: int) -> np.ndarray:
    """Align discrete labels without reading a future source sample."""

    values = np.asarray(labels).reshape(-1)
    idx = np.floor(np.arange(target_len, dtype=np.float64) / target_fs * source_fs + 1e-12).astype(np.int64)
    idx = np.clip(idx, 0, values.size - 1)
    return values[idx].astype(np.int16)


def _detect_wesad_rpeaks(ecg: np.ndarray, fs: float, cache_path: Path | None = None) -> np.ndarray:
    """Detect WESAD R peaks with NeuroKit's ECG pipeline and cache indices."""

    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)
    try:
        import neurokit2 as nk
    except ImportError as exc:  # pragma: no cover
        raise ImportError("v3 WESAD preprocessing requires neurokit2") from exc
    clean = nk.ecg_clean(np.asarray(ecg, dtype=np.float64).reshape(-1), sampling_rate=fs, method="neurokit")
    _, info = nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit", correct_artifacts=True)
    peaks = np.asarray(info["ECG_R_Peaks"], dtype=np.int64)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, peaks)
    return peaks


def future_hr_from_peaks(rpeak_times: np.ndarray, start_sec: float, end_sec: float) -> float:
    """Compute mean HR from complete RR intervals whose two peaks lie in the future horizon."""

    peaks = np.asarray(rpeak_times, dtype=np.float64)
    selected = peaks[(peaks >= start_sec) & (peaks <= end_sec)]
    if selected.size < 2:
        return float("nan")
    rr = np.diff(selected)
    rr = rr[(rr >= 0.3) & (rr <= 2.0)]
    if rr.size == 0:
        return float("nan")
    return float(60.0 / np.mean(rr))


def label_only_pat(
    ppg: np.ndarray,
    rpeaks: np.ndarray,
    fs: float,
    min_delay: float = 0.08,
    max_delay: float = 0.45,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure R-to-maximum-upstroke PAT from raw-rate PPG for label use only."""

    ppg = _forward_fill_finite(ppg)
    sos = signal.butter(4, [0.5, 8.0], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, ppg)
    centered = filtered - np.median(filtered)
    sign = 1.0 if np.quantile(centered, 0.995) >= -np.quantile(centered, 0.005) else -1.0
    filtered = sign * centered
    derivative = np.gradient(filtered) * float(fs)
    beat_times: list[float] = []
    values: list[float] = []
    lo_offset = int(round(min_delay * fs))
    hi_offset = int(round(max_delay * fs))
    for rpeak in np.asarray(rpeaks, dtype=np.int64):
        lo = int(rpeak) + lo_offset
        hi = min(int(rpeak) + hi_offset, filtered.size - 1)
        if lo >= hi:
            continue
        pulse_peak = lo + int(np.argmax(filtered[lo : hi + 1]))
        if pulse_peak <= lo:
            continue
        upstroke = lo + int(np.argmax(derivative[lo : pulse_peak + 1]))
        delay = (upstroke - int(rpeak)) / float(fs)
        if min_delay <= delay <= max_delay:
            beat_times.append(int(rpeak) / float(fs))
            values.append(delay)
    return np.asarray(beat_times, dtype=np.float64), np.asarray(values, dtype=np.float32)


def future_pat(
    beat_times: np.ndarray | None,
    pat_values: np.ndarray | None,
    start_sec: float,
    end_sec: float,
) -> float:
    """Return the median future PAT for one forecast horizon."""

    if beat_times is None or pat_values is None:
        return float("nan")
    mask = (beat_times >= start_sec) & (beat_times <= end_sec) & np.isfinite(pat_values)
    return float(np.median(pat_values[mask])) if mask.any() else float("nan")


def _activity_summary(activity: np.ndarray | None, start: int, end: int, horizon_end: int) -> tuple[int, int, bool]:
    """Return input mode, future mode, and strict no-boundary-crossing flag."""

    if activity is None:
        return -1, -1, True
    current = np.asarray(activity[start:end], dtype=np.int64)
    future = np.asarray(activity[end:horizon_end], dtype=np.int64)
    if current.size == 0 or future.size == 0:
        return -1, -1, False
    cur_mode = int(Counter(current.tolist()).most_common(1)[0][0])
    fut_mode = int(Counter(future.tolist()).most_common(1)[0][0])
    pure = bool(np.all(current == current[0]) and np.all(future == future[0]) and current[0] == future[0])
    return cur_mode, fut_mode, pure


def _split_map(splits: SplitSubjects) -> dict[str, str]:
    return {
        **{subject: "train" for subject in splits.train},
        **{subject: "val" for subject in splits.val},
        **{subject: "test" for subject in splits.test},
    }


def _starts(length: int, input_steps: int, horizon_steps: int, stride_steps: int) -> range:
    last = length - input_steps - horizon_steps
    return range(0, max(-1, last) + 1, stride_steps) if last >= 0 else range(0)


def _discover_dalia(raw_dir: Path) -> list[Path]:
    paths = sorted(raw_dir.glob("S*/S*.pkl"), key=lambda p: natural_key(p.parent.name))
    expected = {f"S{i}" for i in range(1, 16)}
    found = {p.parent.name for p in paths}
    if found != expected:
        raise RuntimeError(f"PPG-DaLiA must contain all 15 subjects; missing={sorted(expected-found, key=natural_key)}")
    for path in paths:
        try:
            with path.open("rb") as handle:
                pickle.load(handle, encoding="latin1")
        except Exception as exc:
            raise RuntimeError(f"Unreadable PPG-DaLiA pickle: {path}: {exc}") from exc
    return paths


def _parse_ptt_subject(record_name: str) -> str:
    return record_name.split("_", maxsplit=1)[0].upper()


def _duration_and_subjects(cfg: Mapping[str, Any]) -> tuple[list[Path], dict[str, int], list[str]]:
    """Discover records and target-grid lengths without changing split membership."""

    name = str(cfg["name"])
    raw_dir = Path(str(cfg["raw_dir"]))
    target_fs = float(cfg["target_fs"])
    if name == "wesad":
        paths = discover_wesad_pickles(raw_dir)
        lengths = {}
        for path in paths:
            raw = _load_wesad_pickle(path)
            lengths[path.parent.name] = int(math.floor(np.asarray(raw["label"]).size / float(cfg.get("label_fs", 700)) * target_fs))
        return paths, lengths, [p.parent.name for p in paths]
    if name == "ppg_dalia":
        paths = _discover_dalia(raw_dir)
        lengths = {}
        for path in paths:
            raw = _load_wesad_pickle(path)
            lengths[path.parent.name] = int(math.floor(np.asarray(raw["signal"]["wrist"]["BVP"]).size / 64.0 * target_fs))
        return paths, lengths, [p.parent.name for p in paths]
    if name == "ptt_ppg":
        paths = discover_ptt_records(raw_dir)
        lengths = {}
        for path in paths:
            fs, sig_len, _ = _read_ptt_header_info(path)
            lengths[path.name] = int(math.floor(sig_len / fs * target_fs))
        subjects = sorted({_parse_ptt_subject(p.name) for p in paths}, key=natural_key)
        return paths, lengths, subjects
    raise KeyError(f"Unknown v3 dataset {name!r}")


def _load_wesad(path: Path, cfg: Mapping[str, Any], target_len: int) -> V3Record:
    raw = _load_wesad_pickle(path)
    target_fs = float(cfg["target_fs"])
    aligned = []
    for node in cfg["nodes"]:
        values = _extract_wesad_node(raw, node)
        aligned.append(causal_align_signal(values, float(node["fs"]), target_fs, target_len, FILTER_BANDS[str(node.get("filter", "none"))]))
    subject = path.parent.name
    ecg = _extract_wesad_node(raw, next(node for node in cfg["nodes"] if str(node["name"]) == "ecg"))
    cache_path = Path(str(cfg.get("rpeak_cache_dir", "data/cache_v3/wesad_rpeaks"))) / f"{subject}.npy"
    rpeaks = _detect_wesad_rpeaks(ecg, float(cfg.get("label_fs", 700)), cache_path)
    return V3Record(subject, subject, np.stack(aligned), rpeaks / float(cfg.get("label_fs", 700)))


def _load_dalia(path: Path, cfg: Mapping[str, Any], target_len: int) -> V3Record:
    raw = _load_wesad_pickle(path)
    target_fs = float(cfg["target_fs"])
    aligned = []
    for node in cfg["nodes"]:
        values = _extract_wesad_node(raw, node)
        aligned.append(causal_align_signal(values, float(node["fs"]), target_fs, target_len, FILTER_BANDS[str(node.get("filter", "none"))]))
    subject = path.parent.name
    rpeaks = np.asarray(raw["rpeaks"], dtype=np.float64) / 700.0
    activity = previous_hold_labels(np.asarray(raw["activity"]), float(cfg.get("activity_fs", 4)), target_fs, target_len)
    return V3Record(subject, subject, np.stack(aligned), rpeaks, activity=activity)


def _load_ptt(path: Path, cfg: Mapping[str, Any], target_len: int) -> V3Record:
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("v3 PTT-PPG preprocessing requires wfdb") from exc
    record = wfdb.rdrecord(str(path))
    source_fs = float(record.fs)
    names = list(record.sig_name)
    raw = np.asarray(record.p_signal, dtype=np.float32)
    aligned = []
    for node in cfg["nodes"]:
        key = str(node["key"])
        if key not in names:
            raise KeyError(f"{key!r} missing from {path.name}; available={names}")
        aligned.append(causal_align_signal(raw[:, names.index(key)], source_fs, float(cfg["target_fs"]), target_len, FILTER_BANDS[str(node.get("filter", "none"))]))
    ann = wfdb.rdann(str(path), "atr")
    rpeaks = np.asarray(ann.sample, dtype=np.int64)
    ir_key = str(cfg.get("pat_ppg_key", "pleth_2"))
    pat_times, pat_values = label_only_pat(raw[:, names.index(ir_key)], rpeaks, source_fs)
    _, _, info = _read_ptt_header_info(path)
    activity_name = str(info.get("activity", "unknown")).lower()
    activity_code = {"sit": 0, "walk": 1, "run": 2}.get(activity_name, -1)
    activity = np.full(target_len, activity_code, dtype=np.int16)
    subject = _parse_ptt_subject(path.name)
    return V3Record(subject, path.name, np.stack(aligned), rpeaks / source_fs, activity, pat_times, pat_values)


def _open_arrays(
    output_dir: Path,
    split_counts: Mapping[str, int],
    n_nodes: int,
    input_steps: int,
    dtype: str,
    include_pat: bool,
) -> dict[str, dict[str, np.ndarray]]:
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split, count in split_counts.items():
        folder = output_dir / split
        folder.mkdir(parents=True, exist_ok=True)
        current = {
            "X": np.lib.format.open_memmap(folder / "X.npy", "w+", dtype=dtype, shape=(count, n_nodes, input_steps, 1)),
            "y_hr": np.lib.format.open_memmap(folder / "y_hr.npy", "w+", dtype=dtype, shape=(count, 1)),
            "subject": np.lib.format.open_memmap(folder / "subject.npy", "w+", dtype="U16", shape=(count,)),
            "record": np.lib.format.open_memmap(folder / "record.npy", "w+", dtype="U48", shape=(count,)),
            "start_idx": np.lib.format.open_memmap(folder / "start_idx.npy", "w+", dtype=np.int64, shape=(count,)),
            "current_hr": np.lib.format.open_memmap(folder / "current_hr.npy", "w+", dtype=dtype, shape=(count,)),
            "activity": np.lib.format.open_memmap(folder / "activity.npy", "w+", dtype=np.int16, shape=(count,)),
            "future_activity": np.lib.format.open_memmap(folder / "future_activity.npy", "w+", dtype=np.int16, shape=(count,)),
            "activity_valid": np.lib.format.open_memmap(folder / "activity_valid.npy", "w+", dtype=np.bool_, shape=(count,)),
        }
        if include_pat:
            current["y_pat"] = np.lib.format.open_memmap(folder / "y_pat.npy", "w+", dtype=dtype, shape=(count, 1))
            current["current_pat"] = np.lib.format.open_memmap(folder / "current_pat.npy", "w+", dtype=dtype, shape=(count,))
        arrays[split] = current
    return arrays


def preprocess_v3(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Create one leakage-safe processed dataset from a plain config mapping."""

    target_fs = float(cfg.get("target_fs", 64))
    input_steps = int(round(float(cfg.get("input_sec", 60)) * target_fs))
    horizon_steps = int(round(float(cfg.get("horizon_sec", 8)) * target_fs))
    stride_steps = int(round(float(cfg.get("stride_sec", 5)) * target_fs))
    output_dir = Path(str(cfg["processed_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, lengths, subjects = _duration_and_subjects(cfg)
    split_cfg = cfg.get("subject_split", {"train": 0.6, "val": 0.15, "test": 0.25, "seed": 42})
    splits = subject_disjoint_split(subjects, float(split_cfg["train"]), float(split_cfg["val"]), float(split_cfg["test"]), int(split_cfg["seed"]))
    subject_to_split = _split_map(splits)
    counts = {"train": 0, "val": 0, "test": 0}
    for path in paths:
        key = path.name if str(cfg["name"]) == "ptt_ppg" else path.parent.name
        subject = _parse_ptt_subject(path.name) if str(cfg["name"]) == "ptt_ppg" else path.parent.name
        counts[subject_to_split[subject]] += len(_starts(lengths[key], input_steps, horizon_steps, stride_steps))
    node_names = [str(node["name"]) for node in cfg["nodes"]]
    include_pat = str(cfg["name"]) == "ptt_ppg"
    arrays = _open_arrays(output_dir, counts, len(node_names), input_steps, str(cfg.get("output_dtype", "float32")), include_pat)
    positions = {split: 0 for split in counts}
    valid_hr = {split: 0 for split in counts}
    valid_pat = {split: 0 for split in counts}
    pure_activity = {split: 0 for split in counts}
    loaders = {"wesad": _load_wesad, "ppg_dalia": _load_dalia, "ptt_ppg": _load_ptt}
    for path in tqdm(paths, desc=f"v3 {cfg['name']}"):
        key = path.name if str(cfg["name"]) == "ptt_ppg" else path.parent.name
        record = loaders[str(cfg["name"])](path, cfg, lengths[key])
        split = subject_to_split[record.subject]
        for start in _starts(record.signals.shape[1], input_steps, horizon_steps, stride_steps):
            idx = positions[split]
            end = start + input_steps
            horizon_end = end + horizon_steps
            start_sec = end / target_fs
            end_sec = horizon_end / target_fs
            input_start_sec = start / target_fs
            x = causal_window_zscore(record.signals[:, start:end])
            hr = future_hr_from_peaks(record.rpeak_times, start_sec, end_sec)
            activity, future_activity, activity_valid = _activity_summary(record.activity, start, end, horizon_end)
            arrays[split]["X"][idx, ..., 0] = x
            arrays[split]["y_hr"][idx, 0] = hr
            arrays[split]["subject"][idx] = record.subject
            arrays[split]["record"][idx] = record.record
            arrays[split]["start_idx"][idx] = start
            arrays[split]["current_hr"][idx] = future_hr_from_peaks(
                record.rpeak_times, input_start_sec, start_sec
            )
            arrays[split]["activity"][idx] = activity
            arrays[split]["future_activity"][idx] = future_activity
            arrays[split]["activity_valid"][idx] = activity_valid
            if np.isfinite(hr):
                valid_hr[split] += 1
            if activity_valid:
                pure_activity[split] += 1
            if include_pat:
                pat = future_pat(record.pat_rpeak_times, record.pat_values, start_sec, end_sec)
                arrays[split]["y_pat"][idx, 0] = pat
                arrays[split]["current_pat"][idx] = future_pat(
                    record.pat_rpeak_times, record.pat_values, input_start_sec, start_sec
                )
                if np.isfinite(pat):
                    valid_pat[split] += 1
            positions[split] += 1
    for split_arrays in arrays.values():
        for array in split_arrays.values():
            array.flush()
    if positions != counts:
        raise RuntimeError(f"Write-count mismatch: {positions} != {counts}")
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    train_hr = np.asarray(arrays["train"]["y_hr"]).reshape(-1)
    metadata = {
        "dataset": str(cfg["name"]),
        "protocol_version": "v3.0",
        "target_fs": target_fs,
        "input_steps": input_steps,
        "horizon_steps": horizon_steps,
        "short_steps": 1,
        "stride_steps": stride_steps,
        "node_names": node_names,
        "splits": {"train": splits.train, "val": splits.val, "test": splits.test},
        "split_counts": counts,
        "valid_hr_counts": valid_hr,
        "valid_pat_counts": valid_pat if include_pat else None,
        "activity_pure_counts": pure_activity,
        "targets": ["hr", "pat"] if include_pat else ["hr"],
        "target_summary": {
            "hr_train_mean": float(np.nanmean(train_hr)),
            "hr_train_std": float(np.nanstd(train_hr)),
        },
        "target_definition": {
            "hr": "60 divided by the mean of valid complete ECG RR intervals wholly contained in the future horizon",
            "pat": "median future ECG R-peak to infrared PPG maximum-upstroke delay" if include_pat else None,
        },
        "preprocessing": {
            "filtering": "causal fourth-order Butterworth SOS with steady-state initialization",
            "resampling": "causal anti-alias low-pass followed by previous-sample alignment",
            "normalization": "per-node z-score using the current input window only",
            "future_information_in_input": False,
            "bp_endpoint_interpolation": False,
        },
        "record_identifiers_saved": True,
        "input_only_baseline_metadata": ["current_hr", "current_pat"] if include_pat else ["current_hr"],
        "hr_prediction_mode": "residual_to_input_period_hr",
        "source_sha256": source_hash,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    metadata = preprocess_v3(_load_yaml(args.config))
    print(json.dumps({"dataset": metadata["dataset"], "split_counts": metadata["split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
