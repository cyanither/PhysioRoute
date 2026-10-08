#!/usr/bin/env python3
"""Generate the reviewer-motivated v3 ablation and control queues."""

from __future__ import annotations

from pathlib import Path


SEEDS = range(42, 47)
SENSITIVITY_SEEDS = range(42, 45)


def row(data: str, seed: int, output: str, overrides: str = "") -> str:
    """Return one run_v3_queue-compatible row."""

    return f"{data}\tpicdgn_v3\t{seed}\t{output}/seed{seed}\t{overrides}\n"


def write(path: str, rows: list[str]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("".join(rows), encoding="utf-8")


def build_controls() -> list[str]:
    variants = {
        "pat_shuffled": ("ptt_ppg_v3_pat_shuffled", ""),
        "time_reversed": ("ptt_ppg_v3_time_reversed", ""),
        "channels_scrambled": ("ptt_ppg_v3_channels_scrambled", ""),
        "pat_only": ("ptt_ppg_v3", "model.loss.lambda_hr=0"),
    }
    return [
        row(data, seed, f"results_v3/controls/ptt_ppg/{name}", overrides)
        for name, (data, overrides) in variants.items()
        for seed in SEEDS
    ]


def build_sensitivity() -> list[str]:
    variants = {
        "tau_max_250ms": "model.tau_max_ms=250",
        "tau_max_750ms": "model.tau_max_ms=750",
        "chunk_1s": "model.chunk_sec=1",
        "chunk_4s": "model.chunk_sec=4",
        "message_steps_2": "model.message_passing_steps=2",
        "no_gumbel_anneal": "model.gumbel_temperature_end=1.0",
    }
    return [
        row("ptt_ppg_v3", seed, f"results_v3/sensitivity/ptt_ppg/{name}", overrides)
        for name, overrides in variants.items()
        for seed in SENSITIVITY_SEEDS
    ]


def build_ptt_ablation() -> list[str]:
    variants = {
        "global_pooled": "model.ablations.graph_type=global_dynamic",
        "static_global": "model.ablations.graph_type=static",
        "learned_static": "model.ablations.graph_type=learned_static",
        "undirected": (
            "model.ablations.directed=false model.ablations.use_direction_prior=false "
            "model.loss.lambda_dir=0"
        ),
        "no_delay": (
            "model.ablations.delay_mode=zero model.ablations.use_pat_prior=false "
            "model.loss.lambda_pat=0"
        ),
        "no_pat_prior": "model.ablations.use_pat_prior=false model.loss.lambda_pat=0",
        "no_sparsity": "model.ablations.use_sparsity=false model.loss.lambda_sp=0",
        "no_direction_prior": "model.ablations.use_direction_prior=false model.loss.lambda_dir=0",
    }
    return [
        row("ptt_ppg_v3", seed, f"results_v3/ablation/ptt_ppg/{name}", overrides)
        for name, overrides in variants.items()
        for seed in SEEDS
    ]


def build_cross_dataset_ablation() -> list[str]:
    variants = {
        "global_pooled": "model.ablations.graph_type=global_dynamic",
        "learned_static": "model.ablations.graph_type=learned_static",
        "undirected": (
            "model.ablations.directed=false model.ablations.use_direction_prior=false "
            "model.loss.lambda_dir=0"
        ),
        "no_delay": (
            "model.ablations.delay_mode=zero model.ablations.use_pat_prior=false "
            "model.loss.lambda_pat=0"
        ),
    }
    rows: list[str] = []
    for dataset in ("wesad", "ppg_dalia"):
        for name, overrides in variants.items():
            for seed in SEEDS:
                rows.append(
                    row(
                        f"{dataset}_v3",
                        seed,
                        f"results_v3/ablation/{dataset}/{name}",
                        overrides,
                    )
                )
    return rows


def main() -> None:
    write("runs/v3/control_ptt_picdgn.tsv", build_controls())
    write("runs/v3/sensitivity_ptt_picdgn.tsv", build_sensitivity())
    write("runs/v3/ablation_ptt_picdgn.tsv", build_ptt_ablation())
    cross_rows = build_cross_dataset_ablation()
    write("runs/v3/ablation_wesad_dalia_picdgn.tsv", cross_rows)
    write("runs/v3/ablation_wesad_picdgn.tsv", [line for line in cross_rows if "ablation/wesad/" in line])
    write("runs/v3/ablation_dalia_picdgn.tsv", [line for line in cross_rows if "ablation/ppg_dalia/" in line])


if __name__ == "__main__":
    main()
