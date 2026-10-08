#!/usr/bin/env python3
"""Build causally filtered, complete-support PAT arrays for v3 PTT-PPG."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy import signal
import wfdb


SOURCE = Path("data/processed_v3/ptt_ppg_h8")
OUTPUT = Path("data/processed_v3/ptt_ppg_h8_causal_pat_candidate")
RAW = Path(
    os.environ.get(
        "PTT_PPG_RAW_DIR",
        "data/raw/ptt_ppg/physionet.org/files/pulse-transit-time-ppg/1.1.0",
    )
)
MIN_DELAY = 0.08
MAX_DELAY = 0.45


def forward_fill(x: np.ndarray) -> np.ndarray:
    """Replace nonfinite samples from past observations only."""

    x = np.asarray(x, dtype=np.float64).reshape(-1).copy()
    finite = np.isfinite(x)
    if finite.all():
        return x
    if not finite.any():
        return np.zeros_like(x)
    first = int(np.flatnonzero(finite)[0])
    x[:first] = 0.0
    indices = np.where(np.isfinite(x), np.arange(x.size), 0)
    np.maximum.accumulate(indices, out=indices)
    return x[indices]


def causal_events(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return R times and PAT values using past-only signal operations."""

    path = RAW / name
    record = wfdb.rdrecord(str(path))
    annotation = wfdb.rdann(str(path), "atr")
    names = list(record.sig_name)
    fs = float(record.fs)
    ppg = forward_fill(np.asarray(record.p_signal[:, names.index("pleth_2")]))
    sos = signal.butter(4, [0.5, 8.0], btype="bandpass", fs=fs, output="sos")
    initial = signal.sosfilt_zi(sos) * float(ppg[0])
    filtered, _ = signal.sosfilt(sos, ppg, zi=initial)
    first_minute = filtered[: min(filtered.size, int(round(60.0 * fs)))]
    centered = first_minute - np.median(first_minute)
    orientation = 1.0 if np.quantile(centered, 0.995) >= -np.quantile(centered, 0.005) else -1.0
    filtered = orientation * filtered
    derivative = np.empty_like(filtered)
    derivative[0] = 0.0
    derivative[1:] = np.diff(filtered) * fs
    lo_offset = int(round(MIN_DELAY * fs))
    hi_offset = int(round(MAX_DELAY * fs))
    beat_times: list[float] = []
    pat_values: list[float] = []
    for rpeak in np.asarray(annotation.sample, dtype=np.int64):
        lo = int(rpeak) + lo_offset
        hi = int(rpeak) + hi_offset
        if lo >= filtered.size or hi >= filtered.size:
            continue
        pulse_peak = lo + int(np.argmax(filtered[lo : hi + 1]))
        if pulse_peak <= lo:
            continue
        upstroke = lo + int(np.argmax(derivative[lo : pulse_peak + 1]))
        delay = (upstroke - int(rpeak)) / fs
        if MIN_DELAY <= delay <= MAX_DELAY:
            beat_times.append(int(rpeak) / fs)
            pat_values.append(delay)
    return np.asarray(beat_times, dtype=np.float64), np.asarray(pat_values, dtype=np.float32)


def interval_median(
    beat_times: np.ndarray,
    pat_values: np.ndarray,
    start: float,
    end: float,
) -> float:
    """Use only R peaks whose entire 80--450 ms search support is in interval."""

    mask = (
        (beat_times >= start)
        & (beat_times + MAX_DELAY <= end)
        & np.isfinite(pat_values)
    )
    return float(np.median(pat_values[mask])) if mask.any() else float("nan")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    metadata = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8"))
    target_fs = float(metadata["target_fs"])
    input_sec = float(metadata["input_steps"]) / target_fs
    horizon_sec = float(metadata["horizon_steps"]) / target_fs
    OUTPUT.mkdir(parents=True)
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    report: dict[str, object] = {
        "definition": (
            "Causal fourth-order 0.5--8 Hz PPG filter; backward derivative; sensor "
            "orientation estimated from the first 60 s; interval inclusion requires "
            "the complete ECG-R+80--450 ms search support inside the interval"
        ),
        "current_pat_role": "external persistence baseline only; omitted from PI-DDGN model inputs",
        "splits": {},
    }
    for split in ("train", "val", "test"):
        folder = SOURCE / split
        records = np.load(folder / "record.npy", mmap_mode="r").astype(str)
        starts = np.load(folder / "start_idx.npy", mmap_mode="r").reshape(-1)
        old_future = np.load(folder / "y_pat.npy", mmap_mode="r").reshape(-1)
        old_current = np.load(folder / "current_pat.npy", mmap_mode="r").reshape(-1)
        future = np.full(old_future.shape, np.nan, dtype=np.float32)
        current = np.full(old_current.shape, np.nan, dtype=np.float32)
        for index, (record, start_idx) in enumerate(zip(records, starts)):
            if record not in cache:
                cache[record] = causal_events(record)
            beat_times, pat_values = cache[record]
            input_start = float(start_idx) / target_fs
            input_end = input_start + input_sec
            future_end = input_end + horizon_sec
            current[index] = interval_median(beat_times, pat_values, input_start, input_end)
            future[index] = interval_median(beat_times, pat_values, input_end, future_end)
        if not np.isfinite(current).all() or not np.isfinite(future).all():
            raise RuntimeError(
                f"{split}: nonfinite current={int((~np.isfinite(current)).sum())}, "
                f"future={int((~np.isfinite(future)).sum())}"
            )
        output_dir = OUTPUT / split
        output_dir.mkdir()
        np.save(output_dir / "current_pat.npy", current)
        np.save(output_dir / "y_pat.npy", future[:, None])
        report["splits"][split] = {
            "windows": int(future.size),
            "changed_current_vs_previous_strict": int(np.count_nonzero(current != old_current)),
            "changed_future_vs_previous_strict": int(np.count_nonzero(future != old_future)),
            "current_pat_quantiles_ms": (np.quantile(current, [0, 0.05, 0.5, 0.95, 1]) * 1000).tolist(),
            "future_pat_quantiles_ms": (np.quantile(future, [0, 0.05, 0.5, 0.95, 1]) * 1000).tolist(),
            "current_pat_sha256": digest(output_dir / "current_pat.npy"),
            "future_pat_sha256": digest(output_dir / "y_pat.npy"),
        }
    (OUTPUT / "causal_pat_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
