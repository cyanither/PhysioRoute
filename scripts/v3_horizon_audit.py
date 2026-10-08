#!/usr/bin/env python3
"""Quantify target smoothing and persistence across the v3 HR horizons."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOTS = {
    "WESAD": {8: Path("data/processed_v3/wesad_h8"), 30: Path("data/processed_v3/wesad_h30")},
    "PPG-DaLiA": {8: Path("data/processed_v3/ppg_dalia_h8"), 30: Path("data/processed_v3/ppg_dalia_h30")},
    "PTT-PPG": {8: Path("data/processed_v3/ptt_ppg_h8"), 30: Path("data/processed_v3/ptt_ppg_h30")},
}


def summarize(root: Path, split: str) -> dict[str, object]:
    """Return label spread, persistence, and within-record adjacent-label correlation."""

    folder = root / split
    target = np.load(folder / "y_hr.npy", mmap_mode="r").reshape(-1).astype(np.float64)
    current = np.load(folder / "current_hr.npy", mmap_mode="r").reshape(-1).astype(np.float64)
    subject = np.load(folder / "subject.npy", mmap_mode="r").astype(str)
    record_path = folder / "record.npy"
    record = np.load(record_path, mmap_mode="r").astype(str) if record_path.exists() else subject
    start = np.load(folder / "start_idx.npy", mmap_mode="r").reshape(-1)
    finite = np.isfinite(target) & np.isfinite(current)
    per_subject = {
        value: float(np.mean(np.abs(current[finite & (subject == value)] - target[finite & (subject == value)])))
        for value in np.unique(subject)
    }
    adjacent_left: list[float] = []
    adjacent_right: list[float] = []
    for value in np.unique(record):
        indices = np.flatnonzero(record == value)
        order = indices[np.argsort(start[indices])]
        if order.size > 1:
            adjacent_left.extend(target[order[:-1]].tolist())
            adjacent_right.extend(target[order[1:]].tolist())
    adjacent_corr = float(np.corrcoef(adjacent_left, adjacent_right)[0, 1])
    return {
        "windows": int(target.size),
        "subjects": int(len(np.unique(subject))),
        "target_mean_bpm": float(np.mean(target)),
        "target_sd_bpm": float(np.std(target, ddof=1)),
        "target_iqr_bpm": float(np.quantile(target, 0.75) - np.quantile(target, 0.25)),
        "adjacent_5s_stride_target_r": adjacent_corr,
        "current_hr_persistence_window_mae_bpm": float(np.mean(np.abs(current[finite] - target[finite]))),
        "current_hr_persistence_subject_macro_mae_bpm": float(np.mean(list(per_subject.values()))),
        "per_subject_persistence_mae_bpm": per_subject,
    }


def main() -> None:
    results = {
        dataset: {
            str(horizon): {split: summarize(root, split) for split in ("train", "val", "test")}
            for horizon, root in roots.items()
        }
        for dataset, roots in ROOTS.items()
    }
    output = Path("results_v3/horizon_target_audit.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    concise = {
        dataset: {
            horizon: {
                "test_sd": round(values["test"]["target_sd_bpm"], 4),
                "test_adj_r": round(values["test"]["adjacent_5s_stride_target_r"], 4),
                "persistence": round(values["test"]["current_hr_persistence_window_mae_bpm"], 4),
            }
            for horizon, values in horizons.items()
        }
        for dataset, horizons in results.items()
    }
    print(json.dumps(concise, indent=2))


if __name__ == "__main__":
    main()
