"""Hydra-driven training entry point for PI-CDGN and baselines."""

from __future__ import annotations

import hashlib
import inspect
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Dataset

import hydra
from src.baselines import BASELINE_REGISTRY
from src.data.loaders import NodeMappedDataset, PhysioWindowDataset, activity_filtered_dataset
from src.models.losses import multitask_loss
from src.models.picdgn import PICDGN

try:
    import wandb
except Exception:  # pragma: no cover
    wandb = None

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set all major RNG seeds."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.use_deterministic_algorithms(True, warn_only=False)


def git_hash() -> str:
    """Return the current git hash if this directory is a git checkout."""

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"



def source_manifest() -> dict[str, str]:
    """Hash source and config files so every checkpoint is traceable without Git."""

    manifest: dict[str, str] = {}
    for root, patterns in ((Path("src"), ("*.py",)), (Path("configs"), ("*.yaml",))):
        for pattern in patterns:
            for path in sorted(root.rglob(pattern)):
                manifest[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def _int_list(value: Any, default: list[int] | None = None) -> list[int]:
    """Convert Hydra list-like values to Python ints."""

    if value is None:
        return default or []
    if isinstance(value, str):
        return [int(part.strip()) for part in value.split(",") if part.strip()]
    return [int(item) for item in value]


def _maybe_activity_filter(dataset: PhysioWindowDataset, split: str, cfg: DictConfig) -> Dataset[dict[str, Any]]:
    """Apply PPG-DaLiA held-out activity filtering when requested."""

    data_cfg = cfg.data
    holdout = data_cfg.get("activity_holdout", data_cfg.get("holdout_activities", None))
    protocol = str(data_cfg.get("protocol", ""))
    if holdout is None and protocol != "cross_activity":
        return dataset
    holdout_values = _int_list(holdout, default=[2, 3, 4])
    if split == "test":
        return activity_filtered_dataset(dataset, include=holdout_values)
    return activity_filtered_dataset(dataset, exclude=holdout_values)


def _load_metadata(processed_dir: Path) -> dict[str, Any]:
    """Read processed metadata JSON."""

    with (processed_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_datasets_and_metadata(cfg: DictConfig) -> tuple[Dataset[dict[str, Any]], Dataset[dict[str, Any]], dict[str, Any]]:
    """Build train/val datasets and model metadata for standard or cross-dataset runs."""

    if str(cfg.data.name).lower() == "cross":
        target_nodes = list(cfg.data.get("common_nodes", ["ecg", "ppg", "acc_x", "acc_y", "acc_z"]))
        target_names = list(cfg.data.get("target_names", ["wave", "hr"]))
        train_parts: list[Dataset[dict[str, Any]]] = []
        val_parts: list[Dataset[dict[str, Any]]] = []
        metadatas: list[dict[str, Any]] = []
        for dataset_name in cfg.data.train_datasets:
            processed_dir = Path(str(cfg.data.processed_roots[str(dataset_name)]))
            metadatas.append(_load_metadata(processed_dir))
            train_parts.append(NodeMappedDataset(PhysioWindowDataset(processed_dir, "train"), target_nodes, target_names))
            val_parts.append(NodeMappedDataset(PhysioWindowDataset(processed_dir, "val"), target_nodes, target_names))
        if not train_parts:
            raise ValueError("cross data config requires at least one train dataset")
        metadata = dict(metadatas[0])
        metadata["dataset"] = "cross"
        metadata["node_names"] = target_nodes
        metadata["targets"] = target_names
        metadata["horizon_steps"] = max(int(item.get("horizon_steps", metadata["horizon_steps"])) for item in metadatas)
        metadata["short_steps"] = max(int(item.get("short_steps", metadata["short_steps"])) for item in metadatas)
        metadata["target_fs"] = float(metadata.get("target_fs", 64.0))
        return ConcatDataset(train_parts), ConcatDataset(val_parts), metadata

    processed_dir = Path(str(cfg.data.processed_dir))
    metadata = _load_metadata(processed_dir)
    train_ds = _maybe_activity_filter(PhysioWindowDataset(processed_dir, "train"), "train", cfg)
    val_ds = _maybe_activity_filter(PhysioWindowDataset(processed_dir, "val"), "val", cfg)
    return train_ds, val_ds, metadata

def move_batch(batch: Mapping[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Move a dataloader batch to device."""

    x = batch["x"].to(device, non_blocking=True)
    targets = {name: value.to(device, non_blocking=True) for name, value in batch["targets"].items()}
    return x, targets


def _initialize_hr_bias(model: torch.nn.Module, metadata: Mapping[str, Any]) -> torch.nn.Module:
    """Initialize the HR intercept to the training-set mean for faster stable optimization."""

    summary = metadata.get("target_summary", {})
    mean = summary.get("hr_train_mean") if isinstance(summary, Mapping) else None
    heads = getattr(model, "heads", None)
    head = getattr(heads, "hr", None)
    if isinstance(head, torch.nn.Linear) and head.bias is not None:
        with torch.no_grad():
            if metadata.get("hr_prediction_mode") == "residual_to_input_period_hr":
                head.weight.zero_()
                head.bias.zero_()
            elif mean is not None:
                head.bias.fill_(float(mean))
    return model


def build_model(cfg: DictConfig, metadata: Mapping[str, Any]) -> torch.nn.Module:
    """Construct PI-CDGN or a baseline from Hydra config."""

    model_name = str(cfg.model.name).lower()
    num_nodes = len(metadata["node_names"])
    short_steps = int(metadata["short_steps"])
    bp_steps = int(metadata.get("horizon_steps", short_steps))
    target_fs = float(metadata["target_fs"])
    if model_name == "picdgn":
        tau_min = float(cfg.model.get("tau_min_ms", 0.0)) / 1000.0
        tau_max = float(cfg.model.get("tau_max_ms", 800.0)) / 1000.0
        delay_init_ms = cfg.model.get("delay_init_ms")
        delay_init = None if delay_init_ms is None else float(delay_init_ms) / 1000.0
        ablations = dict(cfg.model.get("ablations", {}))
        model = PICDGN(
            num_nodes=num_nodes,
            in_channels=1,
            d_model=int(cfg.model.d_model),
            target_fs=target_fs,
            chunk_sec=float(cfg.model.chunk_sec),
            short_steps=short_steps,
            bp_steps=bp_steps,
            message_passing_steps=int(cfg.model.message_passing_steps),
            tau_min=tau_min,
            tau_max=tau_max,
            node_names=list(metadata["node_names"]),
            head_enabled=dict(cfg.model.get("heads", {})),
            graph_type=str(ablations.get("graph_type", "dynamic")),
            directed=bool(ablations.get("directed", True)),
            delay_mode=str(ablations.get("delay_mode", "learned_ptt_prior")),
            delay_init=delay_init,
            use_edge_uncertainty=bool(ablations.get("use_edge_uncertainty", False)),
            temporal_downsample=int(cfg.model.get("temporal_downsample", 8)),
        )
        return _initialize_hr_bias(model, metadata)
    registry_name = {
        "graph_wavenet": "gwn",
    }.get(model_name, model_name)
    if registry_name not in BASELINE_REGISTRY:
        raise KeyError(f"Unknown model '{model_name}'. Available baselines: {sorted(BASELINE_REGISTRY)}")
    cls = BASELINE_REGISTRY[registry_name]
    kwargs = {
        "num_nodes": num_nodes,
        "in_channels": 1,
        "short_steps": short_steps,
        "bp_steps": bp_steps,
    }
    signature = inspect.signature(cls)
    params = signature.parameters
    if "hidden_size" in params and "hidden_size" in cfg.model:
        kwargs["hidden_size"] = int(cfg.model.hidden_size)
    elif "d_model" in params:
        if "d_model" in cfg.model:
            kwargs["d_model"] = int(cfg.model.d_model)
        elif "hidden_size" in cfg.model:
            kwargs["d_model"] = int(cfg.model.hidden_size)
    for key, param in params.items():
        if key in kwargs or key in {"num_nodes", "in_channels", "short_steps", "bp_steps", "hidden_size", "d_model"}:
            continue
        if key not in cfg.model:
            continue
        value = OmegaConf.to_container(cfg.model[key], resolve=True) if OmegaConf.is_config(cfg.model[key]) else cfg.model[key]
        default = param.default
        if isinstance(value, list):
            kwargs[key] = tuple(int(v) for v in value)
        elif isinstance(default, bool):
            kwargs[key] = bool(value)
        elif isinstance(default, int):
            kwargs[key] = int(value)
        else:
            kwargs[key] = value
    return _initialize_hr_bias(cls(**kwargs), metadata)


def epoch_step(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    cfg: DictConfig,
    device: torch.device,
    node_names: list[str],
) -> dict[str, float]:
    """Run one train or validation epoch."""

    train = optimizer is not None
    model.train(train)
    totals: dict[str, float] = {}
    count = 0
    amp_enabled = bool(cfg.amp and device.type == "cuda")
    for batch in loader:
        x, targets = move_batch(batch, device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                if hasattr(model, "graph_learner") and "current_pat" in batch:
                    current_pat = batch["current_pat"].to(device, non_blocking=True)
                    predictions = model(x, current_pat=current_pat)
                else:
                    predictions = model(x)
                if bool(cfg.get("residual_hr", False)) and "hr" in predictions and "current_hr" in batch:
                    current_hr = batch["current_hr"].to(device, non_blocking=True).reshape(-1, 1)
                    predictions["hr"] = predictions["hr"] + current_hr
                graph_regs = None
                graph = predictions.get("graph") if isinstance(predictions, dict) else None
                if graph is not None and hasattr(model, "graph_learner"):
                    ablations = cfg.model.get("ablations", {})
                    graph_regs = model.graph_learner.regularization_losses(
                        graph,
                        node_names=node_names,
                        pat_reference=targets.get("pat"),
                        use_pat_prior=bool(ablations.get("use_pat_prior", False)),
                        use_direction_prior=bool(ablations.get("use_direction_prior", True)),
                        use_sparsity=bool(ablations.get("use_sparsity", True)),
                    )
                loss, parts = multitask_loss(
                    predictions=predictions,
                    targets=targets,
                    graph_regularizers=graph_regs,
                    lambdas=dict(cfg.model.get("loss", {})),
                )
        if train:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.grad_clip_norm))
                optimizer.step()
        batch_size = int(x.shape[0])
        count += batch_size
        for name, value in parts.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach().cpu()) * batch_size
    return {name: value / max(count, 1) for name, value in totals.items()}


@hydra.main(version_base=None, config_path="../configs", config_name="train/default")
def main(cfg: DictConfig) -> None:
    """Train a configured model."""

    set_seed(int(cfg.seed), deterministic=bool(cfg.deterministic))
    output_dir = Path(str(OmegaConf.to_container(cfg, resolve=True)["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    train_ds, val_ds, metadata = build_datasets_and_metadata(cfg)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        handle.write(OmegaConf.to_yaml(cfg, resolve=True))
    (output_dir / "git_hash.txt").write_text(git_hash())
    (output_dir / "source_manifest.json").write_text(json.dumps(source_manifest(), indent=2))

    loader_generator = torch.Generator()
    loader_generator.manual_seed(int(cfg.seed))
    train_loader = DataLoader(train_ds, batch_size=int(cfg.batch_size), shuffle=True, num_workers=int(cfg.num_workers), pin_memory=True, generator=loader_generator)
    val_loader = DataLoader(val_ds, batch_size=int(cfg.batch_size), shuffle=False, num_workers=int(cfg.num_workers), pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, metadata).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(cfg.epochs)))
    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg.amp and device.type == "cuda"))

    writer = SummaryWriter(str(output_dir / "tb")) if SummaryWriter and bool(cfg.tensorboard.enabled) else None
    if wandb is not None and bool(cfg.wandb.enabled):
        wandb.init(project=str(cfg.wandb.project), config=OmegaConf.to_container(cfg, resolve=True), dir=str(output_dir))

    best_val = float("inf")
    patience = int(cfg.patience)
    bad_epochs = 0
    history: list[dict[str, Any]] = []
    node_names = list(metadata["node_names"])
    for epoch in range(1, int(cfg.epochs) + 1):
        temperature = None
        if hasattr(model, "graph_learner"):
            start_temp = float(cfg.model.get("gumbel_temperature_start", 1.0))
            end_temp = float(cfg.model.get("gumbel_temperature_end", start_temp))
            fraction = (epoch - 1) / max(int(cfg.epochs) - 1, 1)
            temperature = start_temp * ((end_temp / max(start_temp, 1e-8)) ** fraction)
            model.graph_learner.set_temperature(temperature)
        train_metrics = epoch_step(model, train_loader, optimizer, scaler, cfg, device, node_names)
        val_metrics = epoch_step(model, val_loader, None, None, cfg, device, node_names)
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": scheduler.get_last_lr()[0], "gumbel_temperature": temperature}
        history.append(record)
        if writer is not None:
            for split, metrics in (("train", train_metrics), ("val", val_metrics)):
                for name, value in metrics.items():
                    writer.add_scalar(f"{split}/{name}", value, epoch)
        if wandb is not None and bool(cfg.wandb.enabled):
            wandb.log({f"train/{k}": v for k, v in train_metrics.items()} | {f"val/{k}": v for k, v in val_metrics.items()} | {"epoch": epoch})
        val_loss = val_metrics.get("total", float("inf"))
        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "metadata": metadata, "cfg": OmegaConf.to_container(cfg, resolve=True)}, output_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"best_val_loss": best_val, "history": history}, handle, indent=2)
    if writer is not None:
        writer.close()
    if wandb is not None and bool(cfg.wandb.enabled):
        wandb.finish()


if __name__ == "__main__":
    main()
