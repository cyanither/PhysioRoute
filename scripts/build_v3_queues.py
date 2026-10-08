"""Build deterministic TSV queues for canonical v3 experiments."""

from pathlib import Path


DATASETS = ("wesad_v3", "ppg_dalia_v3", "ptt_ppg_v3")
SEEDS = tuple(range(42, 47))
MODEL_QUEUES = {
    "gpu4_picdgn.tsv": "picdgn_v3",
    "gpu5_gwn.tsv": "gwn",
    "gpu6_patchtst.tsv": "patchtst",
    "gpu7_lstm.tsv": "lstm",
    "extra_deep_ppg.tsv": "deep_ppg",
    "extra_itransformer.tsv": "itransformer",
    "extra_mtgnn.tsv": "mtgnn",
}


def canonical_name(data_config: str) -> str:
    return data_config.removesuffix("_v3")


def main() -> None:
    root = Path("runs/v3")
    root.mkdir(parents=True, exist_ok=True)
    for filename, model in MODEL_QUEUES.items():
        rows = []
        for data in DATASETS:
            dataset = canonical_name(data)
            for seed in SEEDS:
                output = f"results_v3/canonical/h8/{dataset}/{model}/seed{seed}"
                rows.append("\t".join((data, model, str(seed), output, "")))
        (root / filename).write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
