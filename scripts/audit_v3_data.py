"""Fail-fast audit of every processed dataset used by v3 experiments."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np

from src.data.preprocess_v3 import future_hr_from_peaks


ROOTS = {
    "wesad_h8": Path("data/processed_v3/wesad_h8"),
    "ppg_dalia_h8": Path("data/processed_v3/ppg_dalia_h8"),
    "ptt_ppg_h8": Path("data/processed_v3/ptt_ppg_h8"),
    "wesad_h30": Path("data/processed_v3/wesad_h30"),
    "ppg_dalia_h30": Path("data/processed_v3/ppg_dalia_h30"),
    "ptt_ppg_h30": Path("data/processed_v3/ptt_ppg_h30"),
}


def audit_root(root: Path) -> dict[str, object]:
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["protocol_version"] == "v3.0"
    split_subjects = {name: set(values) for name, values in metadata["splits"].items()}
    assert split_subjects["train"].isdisjoint(split_subjects["val"])
    assert split_subjects["train"].isdisjoint(split_subjects["test"])
    assert split_subjects["val"].isdisjoint(split_subjects["test"])
    assert len(split_subjects["test"]) >= 4
    report: dict[str, object] = {"metadata": metadata, "splits": {}}
    for split in ("train", "val", "test"):
        folder = root / split
        x = np.load(folder / "X.npy", mmap_mode="r")
        hr = np.load(folder / "y_hr.npy", mmap_mode="r").reshape(-1)
        subject = np.load(folder / "subject.npy", mmap_mode="r").astype(str)
        record = np.load(folder / "record.npy", mmap_mode="r").astype(str)
        starts = np.load(folder / "start_idx.npy", mmap_mode="r")
        assert x.shape[0] == hr.size == subject.size == record.size == starts.size == metadata["split_counts"][split]
        assert np.isfinite(hr).all()
        assert np.all((hr >= 30.0) & (hr <= 220.0))
        keys = np.char.add(np.char.add(record, "|"), starts.astype(str))
        assert np.unique(keys).size == keys.size
        sample_idx = np.linspace(0, len(x) - 1, min(512, len(x)), dtype=int)
        sample = np.asarray(x[sample_idx], dtype=np.float32)[..., 0]
        assert np.isfinite(sample).all()
        means = sample.mean(axis=2)
        stds = sample.std(axis=2)
        assert float(np.max(np.abs(means))) < 1e-3
        nonconstant = stds > 1e-4
        assert np.allclose(stds[nonconstant], 1.0, atol=2e-4)
        activity = np.load(folder / "activity.npy", mmap_mode="r")
        future_activity = np.load(folder / "future_activity.npy", mmap_mode="r")
        activity_valid = np.load(folder / "activity_valid.npy", mmap_mode="r").astype(bool)
        assert not np.any(activity_valid & (activity != future_activity))
        split_report: dict[str, object] = {
            "windows": int(len(hr)),
            "subjects": sorted(np.unique(subject).tolist()),
            "records": int(np.unique(record).size),
            "hr_bpm_quantiles": np.quantile(hr, [0, 0.01, 0.5, 0.99, 1]).tolist(),
            "duplicate_record_start_keys": 0,
            "activity_pure_windows": int(activity_valid.sum()),
            "sample_max_abs_channel_mean": float(np.max(np.abs(means))),
            "sample_max_abs_channel_std_error": float(np.max(np.abs(stds[nonconstant] - 1.0))),
        }
        pat_path = folder / "y_pat.npy"
        if pat_path.exists():
            pat = np.load(pat_path, mmap_mode="r").reshape(-1)
            assert np.isfinite(pat).all()
            assert np.all((pat >= 0.08 - 1e-6) & (pat <= 0.45 + 1e-6))
            split_report["pat_ms_quantiles"] = (np.quantile(pat, [0, 0.01, 0.5, 0.99, 1]) * 1000).tolist()
            # The input-interval PAT estimate is an external comparator only.
            # Keeping it under a name unknown to PhysioWindowDataset prevents it
            # from becoming an implicit delay anchor for PI-DDGN.
            assert not (folder / "current_pat.npy").exists()
            baseline_pat_path = folder / "baseline_current_pat.npy"
            assert baseline_pat_path.exists()
            baseline_pat = np.load(baseline_pat_path, mmap_mode="r").reshape(-1)
            assert baseline_pat.size == pat.size
            assert np.isfinite(baseline_pat).all()
            assert np.all((baseline_pat >= 0.08 - 1e-6) & (baseline_pat <= 0.45 + 1e-6))
            split_report["external_current_pat_ms_quantiles"] = (
                np.quantile(baseline_pat, [0, 0.01, 0.5, 0.99, 1]) * 1000
            ).tolist()
        report["splits"][split] = split_report
    if "pat_protocol" in metadata:
        assert metadata["pat_protocol"]["model_visible_current_pat"] is False
        assert metadata["pat_protocol"]["future_information_in_input"] is False
    if metadata.get("horizon_analysis", {}).get("task") == "hr_only":
        assert metadata["targets"] == ["hr"]
        assert metadata["valid_pat_counts"] is None
        for split in ("train", "val", "test"):
            assert not (root / split / "current_pat.npy").exists()
            assert not (root / split / "y_pat.npy").exists()
    return report


def audit_dalia_official_labels() -> dict[str, object]:
    raw_root = Path(
        os.environ.get("PPG_DALIA_RAW_DIR", "data/raw/ppg_dalia/PPG_FieldStudy")
    )
    per_subject: dict[str, dict[str, float]] = {}
    total_abs = 0.0
    total_count = 0
    for subject_id in range(1, 16):
        subject = f"S{subject_id}"
        path = raw_root / subject / f"{subject}.pkl"
        with path.open("rb") as handle:
            raw = pickle.load(handle, encoding="latin1")
        peaks = np.asarray(raw["rpeaks"], dtype=np.float64) / 700.0
        official = np.asarray(raw["label"], dtype=np.float64).reshape(-1)
        derived = np.asarray([future_hr_from_peaks(peaks, 2.0 * idx, 2.0 * idx + 8.0) for idx in range(official.size)])
        valid = np.isfinite(derived) & np.isfinite(official)
        error = derived[valid] - official[valid]
        per_subject[subject] = {
            "n": int(valid.sum()),
            "mae_bpm": float(np.mean(np.abs(error))),
            "bias_bpm": float(np.mean(error)),
            "pearson_r": float(np.corrcoef(derived[valid], official[valid])[0, 1]),
        }
        total_abs += float(np.abs(error).sum())
        total_count += int(valid.sum())
    overall_mae = total_abs / total_count
    assert overall_mae < 0.5
    assert min(item["pearson_r"] for item in per_subject.values()) > 0.98
    return {"overall_mae_bpm": overall_mae, "n": total_count, "per_subject": per_subject}


def main() -> None:
    report = {name: audit_root(root) for name, root in ROOTS.items()}
    report["ppg_dalia_official_label_validation"] = audit_dalia_official_labels()
    output = Path("results_v3/data_audit.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({name: "PASS" for name in report}, indent=2))


if __name__ == "__main__":
    main()
