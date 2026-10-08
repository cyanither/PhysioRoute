#!/usr/bin/env python3
"""Detect suspiciously degenerate or inconsistent v3 prediction artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def read_run(root: Path) -> dict[str, object]:
    """Summarize prediction distribution and identity for one complete run."""

    with np.load(root / "predictions.npz") as archive:
        pred = np.asarray(archive["pred_hr"], dtype=np.float64).reshape(-1)
        target = np.asarray(archive["target_hr"], dtype=np.float64).reshape(-1)
        subject = np.asarray(archive["meta_subject"]).astype(str).reshape(-1)
    config = (root / "config.yaml").read_text(encoding="utf-8")
    seed_line = next(line for line in config.splitlines() if line.startswith("seed:"))
    residual_line = next((line for line in config.splitlines() if line.startswith("residual_hr:")), "")
    corr = float(np.corrcoef(pred, target)[0, 1]) if np.std(pred) > 0 and np.std(target) > 0 else float("nan")
    return {
        "path": str(root),
        "seed": int(seed_line.split(":", 1)[1]),
        "residual_hr": residual_line.endswith("true"),
        "n": int(pred.size),
        "subjects": sorted(np.unique(subject).tolist()),
        "pred_mean": float(pred.mean()),
        "pred_std": float(pred.std()),
        "target_mean": float(target.mean()),
        "target_std": float(target.std()),
        "pearson_r": corr,
        "mae": float(np.mean(np.abs(pred - target))),
        "prediction": pred,
        "target": target,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results_v3"))
    parser.add_argument("--output", type=Path, default=Path("results_v3/prediction_sanity_audit.json"))
    args = parser.parse_args()
    records = [read_run(path.parent) for path in sorted(args.root.rglob("COMPLETE"))]
    groups: dict[tuple[str, ...], list[dict[str, object]]] = defaultdict(list)
    for record in records:
        relative = Path(str(record["path"])).relative_to(args.root)
        group = relative.parts[:-1]
        groups[group].append(record)

    flags: list[dict[str, object]] = []
    for record in records:
        if float(record["pred_std"]) < 0.01 * max(float(record["target_std"]), 1e-6):
            flags.append({"type": "near_constant_prediction", "path": record["path"]})
        if not np.isfinite(float(record["pearson_r"])):
            flags.append({"type": "nonfinite_correlation", "path": record["path"]})
        if float(record["mae"]) > 3.0 * max(float(record["target_std"]), 1e-6):
            flags.append({"type": "mae_exceeds_three_target_sd", "path": record["path"]})
    duplicate_pairs: list[dict[str, object]] = []
    for group, items in groups.items():
        for left_index, left in enumerate(items):
            for right in items[left_index + 1 :]:
                if int(left["n"]) != int(right["n"]):
                    continue
                maximum = float(np.max(np.abs(left["prediction"] - right["prediction"])))
                if maximum == 0.0:
                    duplicate_pairs.append(
                        {
                            "group": "/".join(group),
                            "seed_a": left["seed"],
                            "seed_b": right["seed"],
                            "interpretation": (
                                "Exact cross-seed equality; acceptable for deterministic models only, "
                                "otherwise inspect initialization sensitivity"
                            ),
                        }
                    )
    serializable = []
    for record in records:
        serializable.append({key: value for key, value in record.items() if key not in {"prediction", "target"}})
    report = {
        "status": "PASS" if not flags else "REVIEW",
        "runs": serializable,
        "flags": flags,
        "exact_cross_seed_prediction_pairs": duplicate_pairs,
        "note": "Exact equality may be expected when a baseline converges to a deterministic residual solution.",
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "flags": len(flags), "exact_pairs": len(duplicate_pairs)}))


if __name__ == "__main__":
    main()
