#!/usr/bin/env python3
"""Aggregate COMPLETE non-canonical v3 protocols by variant and seed."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    """Prefer a repository-relative path in public result summaries."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def record(run: Path) -> dict[str, object]:
    """Read one completed run's resolved identity and test metrics."""

    config = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((run / "test_metrics.json").read_text(encoding="utf-8"))
    return {
        "path": portable_path(run),
        "seed": int(config["seed"]),
        "data": str(config["data"]["name"]),
        "model": str(config["model"]["name"]),
        "residual_hr": bool(config.get("residual_hr", False)),
        "hr_mae": float(metrics["hr_mae"]),
        "hr_subject_macro_mae": float(metrics["hr_subject_macro_mae"]),
        "pat_mae_ms": float(metrics["pat_mae_ms"]) if "pat_mae_ms" in metrics else None,
    }


def summarize(items: list[dict[str, object]]) -> dict[str, object]:
    """Return seed-level mean and sample SD for one variant."""

    items.sort(key=lambda item: int(item["seed"]))
    hr = np.asarray([item["hr_mae"] for item in items], dtype=np.float64)
    macro = np.asarray([item["hr_subject_macro_mae"] for item in items], dtype=np.float64)
    pat = np.asarray([item["pat_mae_ms"] for item in items if item["pat_mae_ms"] is not None])
    return {
        "n_seeds": len(items),
        "seeds": [int(item["seed"]) for item in items],
        "hr_mae_mean": float(hr.mean()),
        "hr_mae_sd": float(hr.std(ddof=1)) if hr.size > 1 else None,
        "hr_subject_macro_mae_mean": float(macro.mean()),
        "hr_subject_macro_mae_sd": float(macro.std(ddof=1)) if macro.size > 1 else None,
        "pat_mae_ms_mean": float(pat.mean()) if pat.size else None,
        "pat_mae_ms_sd": float(pat.std(ddof=1)) if pat.size > 1 else None,
        "runs": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results_v3"))
    parser.add_argument("--output", type=Path, default=Path("results_v3/supplemental_summary.json"))
    args = parser.parse_args()
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    excluded = {"canonical", "superseded"}
    for complete in sorted(args.root.rglob("COMPLETE")):
        relative = complete.parent.relative_to(args.root)
        if relative.parts[0] in excluded:
            continue
        key = "/".join(relative.parts[:-1])
        grouped[key].append(record(complete.parent))
    report = {key: summarize(items) for key, items in sorted(grouped.items())}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                key: {
                    "n": value["n_seeds"],
                    "hr": round(float(value["hr_mae_mean"]), 4),
                    "pat_ms": value["pat_mae_ms_mean"],
                }
                for key, value in report.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
