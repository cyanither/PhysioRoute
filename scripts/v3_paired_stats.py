#!/usr/bin/env python3
"""Subject-level paired inference for complete canonical v3 runs."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from aggregate_v3_results import run_record


def paired_subject_inference(
    a_runs: list[dict[str, object]],
    b_runs: list[dict[str, object]],
    seed: int = 2026,
    draws: int = 50000,
) -> dict[str, object]:
    """Compare subject MAEs after averaging over matched initialization seeds."""

    a_by_seed = {int(run["seed"]): run for run in a_runs}
    b_by_seed = {int(run["seed"]): run for run in b_runs}
    matched_seeds = sorted(set(a_by_seed) & set(b_by_seed))
    if not matched_seeds:
        raise ValueError("No matched seeds for paired comparison")
    subjects = sorted(
        set.intersection(
            *[
                set(a_by_seed[value]["per_subject_hr_mae"])
                & set(b_by_seed[value]["per_subject_hr_mae"])
                for value in matched_seeds
            ]
        )
    )
    differences = np.asarray(
        [
            np.mean(
                [
                    float(a_by_seed[value]["per_subject_hr_mae"][subject])
                    - float(b_by_seed[value]["per_subject_hr_mae"][subject])
                    for value in matched_seeds
                ]
            )
            for subject in subjects
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    sampled = rng.choice(differences, size=(draws, len(differences)), replace=True).mean(axis=1)
    observed = abs(float(differences.mean()))
    null = np.asarray(
        [
            float(np.mean(differences * np.asarray(signs)))
            for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
        ]
    )
    return {
        "estimand": "model A minus model B subject-level MAE, averaged over matched seeds",
        "matched_seeds": matched_seeds,
        "n_subjects": len(subjects),
        "subjects": subjects,
        "per_subject_difference_bpm": {
            subject: float(value) for subject, value in zip(subjects, differences)
        },
        "mean_difference_bpm": float(differences.mean()),
        "bootstrap_ci95_low_bpm": float(np.quantile(sampled, 0.025)),
        "bootstrap_ci95_high_bpm": float(np.quantile(sampled, 0.975)),
        "exact_two_sided_signflip_p": float(np.mean(np.abs(null) >= observed - 1e-12)),
    }


def benjamini_hochberg(p_values: dict[str, float]) -> dict[str, float]:
    """Return monotone Benjamini--Hochberg adjusted p-values."""

    ordered = sorted(p_values, key=p_values.get)
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank_from_end, key in enumerate(reversed(ordered), start=1):
        rank = count - rank_from_end + 1
        running = min(running, p_values[key] * count / rank)
        adjusted[key] = float(min(running, 1.0))
    return adjusted


def collect(root: Path) -> dict[tuple[str, str], list[dict[str, object]]]:
    """Collect only COMPLETE runs under dataset/model/seed directories."""

    groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for complete in sorted(root.rglob("COMPLETE")):
        relative = complete.parent.relative_to(root)
        if len(relative.parts) != 3:
            continue
        dataset, model, _ = relative.parts
        groups[(dataset, model)].append(run_record(complete.parent))
    return groups


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results_v3/canonical/h8"))
    parser.add_argument("--reference-model", default="picdgn_v3")
    parser.add_argument("--simple-baselines", type=Path, default=Path("results_v3/simple_baselines.json"))
    parser.add_argument("--output", type=Path, default=Path("results_v3/canonical_h8_paired_stats.json"))
    args = parser.parse_args()
    groups = collect(args.root)
    simple = json.loads(args.simple_baselines.read_text(encoding="utf-8"))
    baseline_names = {"wesad": "WESAD", "ppg_dalia": "PPG-DaLiA", "ptt_ppg": "PTT-PPG"}
    report: dict[str, object] = {}
    datasets = sorted({dataset for dataset, _ in groups})
    for dataset in datasets:
        reference = groups.get((dataset, args.reference_model), [])
        if not reference:
            continue
        persistence_subject = simple[baseline_names[dataset]]["current_hr_persistence"]["per_subject_mae"]
        persistence_runs = [
            {"seed": int(run["seed"]), "per_subject_hr_mae": persistence_subject}
            for run in reference
        ]
        persistence_key = f"{dataset}/{args.reference_model}-minus-current_hr_persistence"
        report[persistence_key] = paired_subject_inference(reference, persistence_runs)
        for current_dataset, model in sorted(groups):
            if current_dataset != dataset or model == args.reference_model:
                continue
            key = f"{dataset}/{args.reference_model}-minus-{model}"
            report[key] = paired_subject_inference(reference, groups[(dataset, model)])
    adjusted = benjamini_hochberg(
        {key: float(value["exact_two_sided_signflip_p"]) for key, value in report.items()}
    ) if report else {}
    for key, value in report.items():
        value["bh_adjusted_p_across_reported_model_comparisons"] = adjusted[key]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value["n_subjects"] for key, value in report.items()}, indent=2))


if __name__ == "__main__":
    main()
