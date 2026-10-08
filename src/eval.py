"""Evaluation metrics and checkpoint evaluation entry point."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
try:
    import hydra
    from omegaconf import DictConfig
except Exception:  # pragma: no cover
    hydra = None
    DictConfig = Any
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.loaders import NodeMappedDataset, PhysioWindowDataset, activity_filtered_dataset


def waveform_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute waveform MSE and Pearson correlation."""

    mask = np.isfinite(target)
    if not mask.any():
        return {"mse": float("nan"), "pearson_r": float("nan")}
    mse = float(np.mean((pred[mask] - target[mask]) ** 2))
    if pred[mask].size < 2:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(pred[mask].reshape(-1), target[mask].reshape(-1))[0, 1])
    return {"mse": mse, "pearson_r": corr}


def regression_metrics(pred: np.ndarray, target: np.ndarray, prefix: str) -> dict[str, float]:
    """Compute MAE, MAPE, and R2 for scalar regression."""

    pred = pred.reshape(-1)
    target = target.reshape(-1)
    mask = np.isfinite(target)
    if not mask.any():
        return {f"{prefix}_mae": float("nan"), f"{prefix}_mape": float("nan"), f"{prefix}_r2": float("nan")}
    error = pred[mask] - target[mask]
    mae = float(np.mean(np.abs(error)))
    denom = np.clip(np.abs(target[mask]), 1e-6, None)
    mape = float(np.mean(np.abs(error) / denom) * 100.0)
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((target[mask] - np.mean(target[mask])) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {f"{prefix}_mae": mae, f"{prefix}_mape": mape, f"{prefix}_r2": r2}


def bp_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Compute BP MAE and threshold accuracies for SBP/DBP."""

    metrics: dict[str, float] = {}
    for idx, name in enumerate(("sbp", "dbp")):
        p = pred[..., idx].reshape(-1)
        t = target[..., idx].reshape(-1)
        mask = np.isfinite(t)
        if not mask.any():
            metrics[f"{name}_mae"] = float("nan")
            metrics[f"{name}_acc_5"] = float("nan")
            metrics[f"{name}_acc_10"] = float("nan")
            continue
        err = np.abs(p[mask] - t[mask])
        metrics[f"{name}_mae"] = float(np.mean(err))
        metrics[f"{name}_acc_5"] = float(np.mean(err <= 5.0))
        metrics[f"{name}_acc_10"] = float(np.mean(err <= 10.0))
    return metrics


def binary_metrics(logits: np.ndarray, target: np.ndarray, prefix: str = "stress") -> dict[str, float]:
    """Compute AUROC, AUPRC, and F1 for binary logits."""

    y = target.reshape(-1)
    mask = np.isfinite(y)
    if not mask.any() or len(np.unique(y[mask])) < 2:
        return {f"{prefix}_auroc": float("nan"), f"{prefix}_auprc": float("nan"), f"{prefix}_f1": float("nan")}
    prob = 1.0 / (1.0 + np.exp(-logits.reshape(-1)[mask]))
    truth = y[mask].astype(int)
    return {
        f"{prefix}_auroc": float(roc_auc_score(truth, prob)),
        f"{prefix}_auprc": float(average_precision_score(truth, prob)),
        f"{prefix}_f1": float(f1_score(truth, prob >= 0.5)),
    }


def expected_calibration_error(prob: np.ndarray, target: np.ndarray, bins: int = 15) -> float:
    """Compute binary expected calibration error."""

    y = target.reshape(-1)
    p = prob.reshape(-1)
    mask = np.isfinite(y)
    if not mask.any():
        return float("nan")
    y = y[mask]
    p = p[mask]
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        bucket = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if bucket.any():
            ece += float(bucket.mean() * abs(p[bucket].mean() - y[bucket].mean()))
    return ece


def pat_metrics(pred_delay: np.ndarray, target_pat: np.ndarray) -> dict[str, float]:
    """Compute pulse-arrival-time recovery MAE in milliseconds."""

    mask = np.isfinite(target_pat)
    if not mask.any():
        return {"pat_mae_ms": float("nan")}
    return {"pat_mae_ms": float(np.mean(np.abs(pred_delay[mask] - target_pat[mask])) * 1000.0)}


def uncertainty_detection_auc(uncertainty: np.ndarray, error: np.ndarray) -> float:
    """AUROC for uncertainty detecting high-error predictions."""

    mask = np.isfinite(uncertainty) & np.isfinite(error)
    if mask.sum() < 4:
        return float("nan")
    threshold = np.nanmedian(error[mask])
    labels = (error[mask] >= threshold).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, uncertainty[mask]))


def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    max_forward_batch: int | None = None,
    metadata_out: dict[str, np.ndarray] | None = None,
    residual_hr: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Collect predictions and targets from a loader.

    PI-CDGN uses causal Transformer attention over long windows, so evaluation
    is chunked by default to avoid failing after a successful training run.
    """

    model.eval()
    preds: dict[str, list[np.ndarray]] = {}
    targets: dict[str, list[np.ndarray]] = {}
    metadata_parts: dict[str, list[np.ndarray]] = {}
    if max_forward_batch is None:
        # Graph models do chunked causal attention; no-grad inference fits a much
        # larger batch than training, so 32 keeps results identical while avoiding
        # the batch-4 evaluation bottleneck on long 60 s windows.
        max_forward_batch = 32 if hasattr(model, "graph_learner") else 128
    amp_enabled = device.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            if metadata_out is not None:
                for name in ("subject", "record", "start_idx", "activity", "future_activity", "activity_valid"):
                    if name in batch:
                        value = batch[name]
                        if isinstance(value, torch.Tensor):
                            array = value.detach().cpu().numpy()
                        else:
                            array = np.asarray(value)
                        metadata_parts.setdefault(name, []).append(array)
            x_all = batch["x"].to(device, non_blocking=True)
            y_all = {name: value.to(device, non_blocking=True) for name, value in batch["targets"].items()}
            for start in range(0, x_all.shape[0], max_forward_batch):
                end = min(start + max_forward_batch, x_all.shape[0])
                x = x_all[start:end]
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    if hasattr(model, "graph_learner") and "current_pat" in batch:
                        current_pat = batch["current_pat"][start:end].to(device, non_blocking=True)
                        out = model(x, current_pat=current_pat)
                    else:
                        out = model(x)
                    if residual_hr and "hr" in out and "current_hr" in batch:
                        current_hr = batch["current_hr"][start:end].to(device, non_blocking=True).reshape(-1, 1)
                        out["hr"] = out["hr"] + current_hr
                for name, value in out.items():
                    if isinstance(value, torch.Tensor):
                        preds.setdefault(name, []).append(value.detach().cpu().numpy())
                for name, value in y_all.items():
                    targets.setdefault(name, []).append(value[start:end].detach().cpu().numpy())
    if metadata_out is not None:
        metadata_out.update({name: np.concatenate(values, axis=0) for name, values in metadata_parts.items()})
    return (
        {name: np.concatenate(values, axis=0) for name, values in preds.items()},
        {name: np.concatenate(values, axis=0) for name, values in targets.items()},
    )


def _int_list(value: Any, default: list[int] | None = None) -> list[int]:
    """Convert Hydra list-like values to Python ints."""

    if value is None:
        return default or []
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]


def build_eval_dataset(cfg: DictConfig) -> PhysioWindowDataset | NodeMappedDataset:
    """Build the evaluation dataset for standard, cross-activity, or cross-dataset runs."""

    if str(cfg.data.name).lower() == "cross":
        target_nodes = list(cfg.data.get("common_nodes", ["ecg", "ppg", "acc_x", "acc_y", "acc_z"]))
        target_names = list(cfg.data.get("target_names", ["wave", "hr"]))
        processed_dir = Path(str(cfg.data.processed_roots[str(cfg.data.test_dataset)]))
        return NodeMappedDataset(PhysioWindowDataset(processed_dir, "test"), target_nodes, target_names)

    dataset = PhysioWindowDataset(cfg.data.processed_dir, "test")
    holdout = cfg.data.get("activity_holdout", cfg.data.get("holdout_activities", None))
    protocol = str(cfg.data.get("protocol", ""))
    if holdout is not None or protocol == "cross_activity":
        return activity_filtered_dataset(dataset, include=_int_list(holdout, default=[2, 3, 4]))
    return dataset


def compute_all_metrics(
    preds: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    groups: Mapping[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Compute window-level metrics and subject-macro metrics when IDs are available."""

    metrics: dict[str, float] = {}
    if "wave" in preds and "wave" in targets:
        metrics |= {f"wave_{k}": v for k, v in waveform_metrics(preds["wave"], targets["wave"]).items()}
    if "hr" in preds and "hr" in targets:
        metrics |= regression_metrics(preds["hr"], targets["hr"], "hr")
    if "bp" in preds and "bp" in targets:
        metrics |= bp_metrics(preds["bp"], targets["bp"])
    if "stress" in preds and "stress" in targets:
        metrics |= binary_metrics(preds["stress"], targets["stress"], "stress")
        prob = 1.0 / (1.0 + np.exp(-preds["stress"]))
        metrics["stress_ece"] = expected_calibration_error(prob, targets["stress"])
    if "pat_delay" in preds and "pat" in targets:
        metrics |= pat_metrics(preds["pat_delay"], targets["pat"])
    if groups is not None and "subject" in groups and "hr" in preds and "hr" in targets:
        subject = np.asarray(groups["subject"]).astype(str).reshape(-1)
        pred = np.asarray(preds["hr"]).reshape(-1)
        truth = np.asarray(targets["hr"]).reshape(-1)
        subject_mae = []
        for value in np.unique(subject):
            mask = (subject == value) & np.isfinite(truth) & np.isfinite(pred)
            if mask.any():
                subject_mae.append(float(np.mean(np.abs(pred[mask] - truth[mask]))))
        metrics["hr_subject_macro_mae"] = float(np.mean(subject_mae)) if subject_mae else float("nan")
        metrics["hr_subject_mae_std"] = float(np.std(subject_mae, ddof=1)) if len(subject_mae) > 1 else float("nan")
        metrics["hr_test_subjects"] = float(len(subject_mae))
    return metrics


if hydra is not None:

    @hydra.main(version_base=None, config_path="../configs", config_name="train/default")
    def main(cfg: DictConfig) -> None:
        """Evaluate ``best.pt`` for the configured run on the test split."""

        from src.train import build_model

        output_dir = Path(str(cfg.output_dir))
        checkpoint_path = output_dir / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        metadata = checkpoint["metadata"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(cfg, metadata).to(device)
        model.load_state_dict(checkpoint["model"])
        dataset = build_eval_dataset(cfg)
        loader = DataLoader(
            dataset,
            batch_size=int(cfg.batch_size),
            shuffle=False,
            num_workers=int(cfg.num_workers),
        )
        sample_metadata: dict[str, np.ndarray] = {}
        preds, targets = collect_predictions(
            model,
            loader,
            device,
            metadata_out=sample_metadata,
            residual_hr=bool(cfg.get("residual_hr", False)),
        )
        metrics = compute_all_metrics(preds, targets, sample_metadata)
        with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)
        np.savez_compressed(
            output_dir / "predictions.npz",
            **{f"pred_{name}": value for name, value in preds.items()},
            **{f"target_{name}": value for name, value in targets.items()},
            **{f"meta_{name}": value for name, value in sample_metadata.items()},
        )
        print(json.dumps(metrics, indent=2))

else:  # pragma: no cover

    def main(_: Any = None) -> None:
        raise RuntimeError("hydra-core is required for the evaluation CLI.")


if __name__ == "__main__":
    main()
