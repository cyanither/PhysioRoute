#!/usr/bin/env python3
"""Recompute v3 PTT-PPG PAT arrays with complete-event interval inclusion.

An event belongs to an input or future interval only when both its ECG R peak
and its matched PPG maximum upstroke occur inside that interval.  The script
writes a separate candidate directory and never overwrites source artifacts.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import wfdb

from src.data.preprocess_v3 import label_only_pat


SOURCE = Path("data/processed_v3/ptt_ppg_h8")
OUTPUT = Path("data/processed_v3/ptt_ppg_h8_strict_pat_candidate")
RAW = Path(
    os.environ.get(
        "PTT_PPG_RAW_DIR",
        "data/raw/ptt_ppg/physionet.org/files/pulse-transit-time-ppg/1.1.0",
    )
)


def digest(path: Path) -> str:
    """Return SHA-256 for one output artifact."""

    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def interval_median(
    beat_times: np.ndarray,
    pat_values: np.ndarray,
    start: float,
    end: float,
) -> float:
    """Median delay for events whose R peak and PPG upstroke are both in interval."""

    upstroke_times = beat_times + pat_values.astype(np.float64)
    mask = (
        (beat_times >= start)
        & (upstroke_times <= end)
        & np.isfinite(pat_values)
    )
    return float(np.median(pat_values[mask])) if mask.any() else float("nan")


def record_events(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct label-only R-to-upstroke events from one raw record."""

    record_path = RAW / name
    record = wfdb.rdrecord(str(record_path))
    annotation = wfdb.rdann(str(record_path), "atr")
    names = list(record.sig_name)
    if "pleth_2" not in names:
        raise KeyError(f"pleth_2 missing from {name}")
    ppg = np.asarray(record.p_signal[:, names.index("pleth_2")], dtype=np.float64)
    return label_only_pat(ppg, np.asarray(annotation.sample, dtype=np.int64), float(record.fs))


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    metadata = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8"))
    fs = float(metadata["target_fs"])
    input_sec = float(metadata["input_steps"]) / fs
    horizon_sec = float(metadata["horizon_steps"]) / fs
    OUTPUT.mkdir(parents=True)
    report: dict[str, object] = {
        "definition": (
            "An interval contains a PAT event only when both the ECG R peak and "
            "matched PPG maximum upstroke lie within its closed boundaries"
        ),
        "splits": {},
    }
    cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
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
                cache[record] = record_events(record)
            beat_times, pat_values = cache[record]
            input_start = float(start_idx) / fs
            input_end = input_start + input_sec
            future_end = input_end + horizon_sec
            current[index] = interval_median(beat_times, pat_values, input_start, input_end)
            future[index] = interval_median(beat_times, pat_values, input_end, future_end)
        if not np.isfinite(current).all() or not np.isfinite(future).all():
            raise RuntimeError(
                f"{split}: nonfinite strict PAT values current={int((~np.isfinite(current)).sum())} "
                f"future={int((~np.isfinite(future)).sum())}"
            )
        split_out = OUTPUT / split
        split_out.mkdir()
        np.save(split_out / "current_pat.npy", current)
        np.save(split_out / "y_pat.npy", future[:, None])
        report["splits"][split] = {
            "windows": int(future.size),
            "changed_current": int(np.count_nonzero(current != old_current)),
            "changed_future": int(np.count_nonzero(future != old_future)),
            "max_abs_current_change_ms": float(np.max(np.abs(current - old_current)) * 1000.0),
            "max_abs_future_change_ms": float(np.max(np.abs(future - old_future)) * 1000.0),
            "current_pat_sha256": digest(split_out / "current_pat.npy"),
            "future_pat_sha256": digest(split_out / "y_pat.npy"),
        }
    (OUTPUT / "strict_pat_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
