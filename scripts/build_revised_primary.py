#!/usr/bin/env python3
"""Build revised-primary summaries without altering completed run artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aggregate_v3_results import run_record
from v3_paired_stats import benjamini_hochberg, paired_subject_inference


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("wesad", "ppg_dalia", "ptt_ppg")
BASELINE_LABELS = {"wesad": "WESAD", "ppg_dalia": "PPG-DaLiA", "ptt_ppg": "PTT-PPG"}


def aggregate_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    window = np.asarray([float(run["window_hr_mae"]) for run in runs])
    macro = np.asarray([float(run["subject_macro_hr_mae"]) for run in runs])
    pat = [float(run["pat_mae_ms"]) for run in runs if run["pat_mae_ms"] is not None]
    return {
        "n_seeds": len(runs),
        "seeds": [int(run["seed"]) for run in runs],
        "window_hr_mae_mean": float(window.mean()),
        "window_hr_mae_sd": float(window.std(ddof=1)),
        "subject_macro_hr_mae_mean": float(macro.mean()),
        "subject_macro_hr_mae_sd": float(macro.std(ddof=1)),
        "pat_mae_ms_mean": float(np.mean(pat)) if pat else None,
        "pat_mae_ms_sd": float(np.std(pat, ddof=1)) if len(pat) > 1 else None,
        "runs": runs,
    }


def main() -> None:
    canonical = json.loads((ROOT / "results_v3/canonical_h8_summary.json").read_text())
    simple = json.loads((ROOT / "results_v3/simple_baselines.json").read_text())
    revised = {
        "selection_rule": (
            "Promote pair-conditioned learned-static PI-DDGN because its mean validation objective "
            "was lower than the chunk-dynamic variant on all three datasets; test artifacts were not altered."
        ),
        "groups": dict(canonical["groups"]),
    }
    stats: dict[str, dict[str, object]] = {}

    for dataset in DATASETS:
        paths = sorted((ROOT / f"results_v3/ablation/{dataset}/learned_static").glob("seed*/COMPLETE"))
        runs = [run_record(path.parent) for path in paths]
        if len(runs) != 5:
            raise RuntimeError(f"Expected five complete learned-static runs for {dataset}, found {len(runs)}")
        runs.sort(key=lambda run: int(run["seed"]))
        revised["groups"][f"{dataset}/picdgn_v3"] = aggregate_runs(runs)

        persistence_subject = simple[BASELINE_LABELS[dataset]]["current_hr_persistence"]["per_subject_mae"]
        persistence_runs = [
            {"seed": int(run["seed"]), "per_subject_hr_mae": persistence_subject} for run in runs
        ]
        stats[f"{dataset}/picdgn_v3-minus-current_hr_persistence"] = paired_subject_inference(
            runs, persistence_runs
        )
        for model in ("deep_ppg", "gwn", "itransformer", "lstm", "mtgnn", "patchtst"):
            baseline_runs = canonical["groups"][f"{dataset}/{model}"]["runs"]
            stats[f"{dataset}/picdgn_v3-minus-{model}"] = paired_subject_inference(runs, baseline_runs)

    adjusted = benjamini_hochberg(
        {key: float(value["exact_two_sided_signflip_p"]) for key, value in stats.items()}
    )
    for key, value in stats.items():
        value["bh_adjusted_p_across_reported_model_comparisons"] = adjusted[key]

    (ROOT / "results_v3/revised_primary_summary.json").write_text(json.dumps(revised, indent=2))
    (ROOT / "results_v3/revised_primary_paired_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({"status": "PASS", "datasets": list(DATASETS)}))


if __name__ == "__main__":
    main()
