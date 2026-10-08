#!/usr/bin/env python3
"""Simple baselines on the strict PPG-DaLiA held-out-activity protocol."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


ROOT = Path("data/processed_v3/ppg_dalia_h8")
HOLDOUT = np.asarray([2, 3, 4], dtype=np.int16)


def selection(split: str, include_holdout: bool) -> np.ndarray:
    """Match the loader's strict activity-valid include/exclude behavior."""

    folder = ROOT / split
    activity = np.load(folder / "activity.npy", mmap_mode="r").reshape(-1)
    valid = np.load(folder / "activity_valid.npy", mmap_mode="r").reshape(-1).astype(bool)
    held = np.isin(activity, HOLDOUT)
    return valid & (held if include_holdout else ~held)


def metrics(prediction: np.ndarray, target: np.ndarray, subject: np.ndarray, activity: np.ndarray) -> dict[str, object]:
    """Return window, subject-macro, and per-activity MAE."""

    prediction = np.asarray(prediction).reshape(-1)
    target = np.asarray(target).reshape(-1)
    subject = np.asarray(subject).astype(str).reshape(-1)
    activity = np.asarray(activity).reshape(-1)
    valid = np.isfinite(prediction) & np.isfinite(target)
    per_subject = {
        value: float(np.mean(np.abs(prediction[valid & (subject == value)] - target[valid & (subject == value)])))
        for value in np.unique(subject)
        if np.any(valid & (subject == value))
    }
    per_activity = {
        str(int(value)): float(np.mean(np.abs(prediction[valid & (activity == value)] - target[valid & (activity == value)])))
        for value in np.unique(activity)
        if np.any(valid & (activity == value))
    }
    return {
        "window_mae": float(np.mean(np.abs(prediction[valid] - target[valid]))),
        "subject_macro_mae": float(np.mean(list(per_subject.values()))),
        "per_subject_mae": per_subject,
        "per_activity_mae": per_activity,
        "windows": int(valid.sum()),
    }


def main() -> None:
    train_mask = selection("train", include_holdout=False)
    test_mask = selection("test", include_holdout=True)
    train_folder = ROOT / "train"
    test_folder = ROOT / "test"
    train_hr = np.load(train_folder / "y_hr.npy", mmap_mode="r").reshape(-1)[train_mask]
    train_current = np.load(train_folder / "current_hr.npy", mmap_mode="r").reshape(-1)[train_mask]
    train_activity = np.load(train_folder / "activity.npy", mmap_mode="r").reshape(-1)[train_mask]
    test_hr = np.load(test_folder / "y_hr.npy", mmap_mode="r").reshape(-1)[test_mask]
    test_current = np.load(test_folder / "current_hr.npy", mmap_mode="r").reshape(-1)[test_mask]
    test_activity = np.load(test_folder / "activity.npy", mmap_mode="r").reshape(-1)[test_mask]
    test_subject = np.load(test_folder / "subject.npy", mmap_mode="r").astype(str)[test_mask]
    median = float(np.nanmedian(train_hr))
    persistence = np.where(np.isfinite(test_current), test_current, median)
    valid_train = np.isfinite(train_hr) & np.isfinite(train_current)
    ar = Ridge(alpha=1.0).fit(
        train_current[valid_train, None],
        train_hr[valid_train] - train_current[valid_train],
    )
    ar_prediction = persistence + ar.predict(persistence[:, None])
    activity_means = {
        int(code): float(np.nanmean(train_hr[train_activity == code]))
        for code in np.unique(train_activity)
        if np.any(train_activity == code)
    }
    activity_prediction = np.asarray([activity_means.get(int(code), median) for code in test_activity])
    results = {
        "protocol": {
            "holdout_activity_codes": HOLDOUT.tolist(),
            "train_windows": int(train_mask.sum()),
            "test_windows": int(test_mask.sum()),
            "boundary_crossing_windows_excluded": True,
        },
        "training_median": metrics(np.full(test_hr.shape, median), test_hr, test_subject, test_activity),
        "current_hr_persistence": metrics(persistence, test_hr, test_subject, test_activity),
        "linear_ar_current_hr": metrics(ar_prediction, test_hr, test_subject, test_activity),
        "training_activity_mean": metrics(activity_prediction, test_hr, test_subject, test_activity),
    }
    output = Path("results_v3/activity_strict_baselines.json")
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
