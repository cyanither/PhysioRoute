#!/usr/bin/env python3
"""Audit integrity and metric reproducibility of every COMPLETE v3 run."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import yaml


REQUIRED = (
    "COMPLETE",
    "best.pt",
    "config.yaml",
    "metrics.json",
    "predictions.npz",
    "run_status.json",
    "source_manifest.json",
    "test_metrics.json",
)


def audit_run(root: Path) -> dict[str, object]:
    """Validate one run and recompute HR/PAT MAEs from saved arrays."""

    missing = [name for name in REQUIRED if not (root / name).is_file() or (root / name).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"{root}: missing/empty files {missing}")
    config = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((root / "test_metrics.json").read_text(encoding="utf-8"))
    status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise RuntimeError(f"{root}: run_status is not complete")
    if not manifest or not all(isinstance(value, str) and len(value) == 64 for value in manifest.values()):
        raise RuntimeError(f"{root}: invalid source manifest")
    with np.load(root / "predictions.npz") as arrays:
        required_arrays = ("pred_hr", "target_hr", "meta_subject")
        absent = [name for name in required_arrays if name not in arrays]
        if absent:
            raise RuntimeError(f"{root}: missing prediction arrays {absent}")
        pred = np.asarray(arrays["pred_hr"]).reshape(-1)
        target = np.asarray(arrays["target_hr"]).reshape(-1)
        subject = np.asarray(arrays["meta_subject"]).astype(str).reshape(-1)
        if not (pred.size == target.size == subject.size and pred.size > 0):
            raise RuntimeError(f"{root}: inconsistent saved prediction lengths")
        valid = np.isfinite(pred) & np.isfinite(target)
        if not valid.all():
            raise RuntimeError(f"{root}: {int((~valid).sum())} nonfinite HR pairs")
        recomputed = float(np.mean(np.abs(pred - target)))
        if not math.isclose(recomputed, float(metrics["hr_mae"]), rel_tol=1e-7, abs_tol=1e-6):
            raise RuntimeError(f"{root}: HR MAE mismatch {recomputed} != {metrics['hr_mae']}")
        subject_values = [
            float(np.mean(np.abs(pred[subject == value] - target[subject == value])))
            for value in np.unique(subject)
        ]
        macro = float(np.mean(subject_values))
        if not math.isclose(macro, float(metrics["hr_subject_macro_mae"]), rel_tol=1e-7, abs_tol=1e-6):
            raise RuntimeError(f"{root}: subject-macro MAE mismatch")
        pat_recomputed = None
        if "pred_pat_delay" in arrays and "target_pat" in arrays:
            pred_pat = np.asarray(arrays["pred_pat_delay"]).reshape(-1)
            target_pat = np.asarray(arrays["target_pat"]).reshape(-1)
            pat_valid = np.isfinite(pred_pat) & np.isfinite(target_pat)
            if pat_valid.any():
                pat_recomputed = float(np.mean(np.abs(pred_pat[pat_valid] - target_pat[pat_valid])) * 1000.0)
                if "pat_mae_ms" not in metrics or not math.isclose(
                    pat_recomputed, float(metrics["pat_mae_ms"]), rel_tol=1e-7, abs_tol=1e-5
                ):
                    raise RuntimeError(f"{root}: PAT MAE mismatch")
    return {
        "path": str(root),
        "data_config": str(config["data"]["name"]),
        "model": str(config["model"]["name"]),
        "seed": int(config["seed"]),
        "windows": int(pred.size),
        "subjects": int(len(np.unique(subject))),
        "hr_mae": recomputed,
        "hr_subject_macro_mae": macro,
        "pat_mae_ms": pat_recomputed,
        "epochs": int(status["epochs_ran"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results_v3"))
    parser.add_argument("--output", type=Path, default=Path("results_v3/run_audit.json"))
    args = parser.parse_args()
    records = [audit_run(path.parent) for path in sorted(args.root.rglob("COMPLETE"))]
    report = {"status": "PASS", "complete_runs": len(records), "runs": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "complete_runs": len(records)}))


if __name__ == "__main__":
    main()
