#!/usr/bin/env python3
"""Build interval-isolated PAT labels and an external current-PAT baseline.

Filtering is performed separately inside each 60-s input interval and each
future interval.  Hence neither calculation can access signal samples across
its own boundary.  The current-period quantity is stored under a name that the
model loader does not recognize, so it remains an evaluation baseline only.
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
OUTPUT = Path("data/processed_v3/ptt_ppg_h8_isolated_pat_candidate")
RAW = Path(
    os.environ.get(
        "PTT_PPG_RAW_DIR",
        "data/raw/ptt_ppg/physionet.org/files/pulse-transit-time-ppg/1.1.0",
    )
)
MAX_DELAY = 0.45


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load_record(name: str) -> tuple[np.ndarray, np.ndarray, float]:
    path = RAW / name
    record = wfdb.rdrecord(str(path))
    annotation = wfdb.rdann(str(path), "atr")
    names = list(record.sig_name)
    return (
        np.asarray(record.p_signal[:, names.index("pleth_2")], dtype=np.float64),
        np.asarray(annotation.sample, dtype=np.int64),
        float(record.fs),
    )


def isolated_pat(
    ppg: np.ndarray,
    rpeaks: np.ndarray,
    fs: float,
    start_sec: float,
    end_sec: float,
) -> float:
    """Calculate PAT using only samples and complete search support in one interval."""

    start = int(round(start_sec * fs))
    stop = int(round(end_sec * fs))
    segment = np.asarray(ppg[start:stop], dtype=np.float64)
    complete = rpeaks[(rpeaks >= start) & (rpeaks + int(round(MAX_DELAY * fs)) < stop)] - start
    if complete.size < 1:
        return float("nan")
    _, values = label_only_pat(segment, complete, fs)
    return float(np.median(values)) if values.size else float("nan")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT}")
    metadata = json.loads((SOURCE / "metadata.json").read_text(encoding="utf-8"))
    target_fs = float(metadata["target_fs"])
    input_sec = float(metadata["input_steps"]) / target_fs
    horizon_sec = float(metadata["horizon_steps"]) / target_fs
    OUTPUT.mkdir(parents=True)
    cache: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    report: dict[str, object] = {
        "definition": (
            "Fourth-order zero-phase 0.5--8 Hz filtering performed independently "
            "inside each interval; interval R peaks require the full 80--450 ms "
            "search support before the interval endpoint; PAT is the median ECG-R "
            "to maximum-PPG-upstroke delay"
        ),
        "future_pat_role": "PI-DDGN training target and held-out diagnostic",
        "current_pat_role": (
            "External persistence baseline only, stored as baseline_current_pat.npy; "
            "not discoverable by PhysioWindowDataset and not supplied to PI-DDGN"
        ),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        folder = SOURCE / split
        records = np.load(folder / "record.npy", mmap_mode="r").astype(str)
        starts = np.load(folder / "start_idx.npy", mmap_mode="r").reshape(-1)
        current = np.full(starts.shape, np.nan, dtype=np.float32)
        future = np.full(starts.shape, np.nan, dtype=np.float32)
        for index, (record, start_idx) in enumerate(zip(records, starts)):
            if record not in cache:
                cache[record] = load_record(record)
            ppg, rpeaks, fs = cache[record]
            input_start = float(start_idx) / target_fs
            input_end = input_start + input_sec
            future_end = input_end + horizon_sec
            current[index] = isolated_pat(ppg, rpeaks, fs, input_start, input_end)
            future[index] = isolated_pat(ppg, rpeaks, fs, input_end, future_end)
        if not np.isfinite(current).all() or not np.isfinite(future).all():
            raise RuntimeError(
                f"{split}: nonfinite current={int((~np.isfinite(current)).sum())}, "
                f"future={int((~np.isfinite(future)).sum())}"
            )
        output_dir = OUTPUT / split
        output_dir.mkdir()
        np.save(output_dir / "baseline_current_pat.npy", current)
        np.save(output_dir / "y_pat.npy", future[:, None])
        report["splits"][split] = {
            "windows": int(future.size),
            "current_pat_quantiles_ms": (np.quantile(current, [0, 0.05, 0.5, 0.95, 1]) * 1000).tolist(),
            "future_pat_quantiles_ms": (np.quantile(future, [0, 0.05, 0.5, 0.95, 1]) * 1000).tolist(),
            "baseline_current_pat_sha256": digest(output_dir / "baseline_current_pat.npy"),
            "future_pat_sha256": digest(output_dir / "y_pat.npy"),
        }
    (OUTPUT / "isolated_pat_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
