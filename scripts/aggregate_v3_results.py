"""Aggregate only complete canonical v3 runs and compute paired subject summaries."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path) -> str:
    """Prefer a repository-relative path in public result summaries."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_record(path: Path) -> dict[str, object]:
    if not (path / "COMPLETE").is_file():
        raise RuntimeError(f"Incomplete run: {path}")
    metrics = json.loads((path / "test_metrics.json").read_text())
    archive = np.load(path / "predictions.npz")
    pred = archive["pred_hr"].reshape(-1)
    truth = archive["target_hr"].reshape(-1)
    subject = archive["meta_subject"].astype(str).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(truth)
    if not valid.all():
        raise RuntimeError(f"Nonfinite prediction/target in {path}: {int((~valid).sum())}")
    per_subject = {
        value: float(np.mean(np.abs(pred[subject == value] - truth[subject == value])))
        for value in np.unique(subject)
    }
    config_lines = (path / "config.yaml").read_text().splitlines()
    seed_line = next(line for line in config_lines if line.startswith("seed:"))
    return {
        "path": portable_path(path),
        "seed": int(seed_line.split(":", 1)[1].strip()),
        "window_hr_mae": float(metrics["hr_mae"]),
        "subject_macro_hr_mae": float(metrics["hr_subject_macro_mae"]),
        "per_subject_hr_mae": per_subject,
        "pat_mae_ms": float(metrics["pat_mae_ms"]) if "pat_mae_ms" in metrics else None,
    }


def bootstrap_paired_subject_difference(
    a: dict[str, float], b: dict[str, float], seed: int = 2026, draws: int = 10000
) -> dict[str, float]:
    subjects = sorted(set(a) & set(b))
    differences = np.asarray([a[subject] - b[subject] for subject in subjects], dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(differences, size=(draws, len(differences)), replace=True).mean(axis=1)
    return {
        "n_subjects": len(subjects),
        "mean_difference_bpm": float(differences.mean()),
        "ci95_low_bpm": float(np.quantile(sampled, 0.025)),
        "ci95_high_bpm": float(np.quantile(sampled, 0.975)),
    }


def aggregate(root: Path) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for complete in sorted(root.rglob("COMPLETE")):
        path = complete.parent
        relative = path.relative_to(root)
        if len(relative.parts) != 3:
            continue
        dataset, model, _ = relative.parts
        grouped[(dataset, model)].append(run_record(path))
    report: dict[str, object] = {"groups": {}, "paired_subject_bootstrap": {}}
    by_dataset: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(dict)
    for (dataset, model), runs in sorted(grouped.items()):
        runs.sort(key=lambda item: int(item["seed"]))
        values = np.asarray([item["window_hr_mae"] for item in runs], dtype=np.float64)
        macro = np.asarray([item["subject_macro_hr_mae"] for item in runs], dtype=np.float64)
        pats = [float(item["pat_mae_ms"]) for item in runs if item["pat_mae_ms"] is not None]
        key = f"{dataset}/{model}"
        report["groups"][key] = {
            "n_seeds": len(runs),
            "seeds": [int(item["seed"]) for item in runs],
            "window_hr_mae_mean": float(values.mean()),
            "window_hr_mae_sd": float(values.std(ddof=1)) if len(values) > 1 else None,
            "subject_macro_hr_mae_mean": float(macro.mean()),
            "subject_macro_hr_mae_sd": float(macro.std(ddof=1)) if len(macro) > 1 else None,
            "pat_mae_ms_mean": float(np.mean(pats)) if pats else None,
            "pat_mae_ms_sd": float(np.std(pats, ddof=1)) if len(pats) > 1 else None,
            "runs": runs,
        }
        by_dataset[dataset][model] = runs
    for dataset, models in sorted(by_dataset.items()):
        if "picdgn_v3" not in models:
            continue
        pi_by_seed = {int(item["seed"]): item for item in models["picdgn_v3"]}
        for model, runs in sorted(models.items()):
            if model == "picdgn_v3":
                continue
            other_by_seed = {int(item["seed"]): item for item in runs}
            comparisons = []
            for seed in sorted(set(pi_by_seed) & set(other_by_seed)):
                comparisons.append(
                    bootstrap_paired_subject_difference(
                        pi_by_seed[seed]["per_subject_hr_mae"],
                        other_by_seed[seed]["per_subject_hr_mae"],
                        seed=2026 + seed,
                    )
                )
            if comparisons:
                report["paired_subject_bootstrap"][f"{dataset}/picdgn_v3-minus-{model}"] = comparisons
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results_v3/canonical/h8"))
    parser.add_argument("--output", type=Path, default=Path("results_v3/canonical_h8_summary.json"))
    args = parser.parse_args()
    report = aggregate(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value["n_seeds"] for key, value in report["groups"].items()}, indent=2))


if __name__ == "__main__":
    main()
