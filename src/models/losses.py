"""Loss functions for PI-CDGN multi-task training."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F


def _finite_mask(target: torch.Tensor) -> torch.Tensor:
    return torch.isfinite(target)


def masked_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error ignoring NaN targets."""

    mask = _finite_mask(target)
    if not mask.any():
        return pred.sum() * 0.0
    return F.mse_loss(pred[mask], target[mask])


def masked_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean absolute error ignoring NaN targets."""

    mask = _finite_mask(target)
    if not mask.any():
        return pred.sum() * 0.0
    return F.l1_loss(pred[mask], target[mask])


def multitask_loss(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    graph_regularizers: Mapping[str, torch.Tensor] | None,
    lambdas: Mapping[str, float],
    edge_alpha: torch.Tensor | None = None,
    edge_beta: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute total configured PI-CDGN loss."""

    device = next(iter(predictions.values())).device
    total = torch.tensor(0.0, device=device)
    parts: dict[str, torch.Tensor] = {}

    if "wave" in predictions and "wave" in targets:
        parts["wave"] = masked_mse(predictions["wave"], targets["wave"])
        total = total + float(lambdas.get("lambda_wave", 1.0)) * parts["wave"]
    if "hr" in predictions and "hr" in targets:
        parts["hr"] = masked_mae(predictions["hr"], targets["hr"])
        total = total + float(lambdas.get("lambda_hr", 1.0)) * parts["hr"]
    if "bp" in predictions and "bp" in targets:
        parts["bp"] = masked_mae(predictions["bp"], targets["bp"])
        total = total + float(lambdas.get("lambda_bp", 1.0)) * parts["bp"]
    if "stress" in predictions and "stress" in targets:
        mask = _finite_mask(targets["stress"])
        if mask.any():
            parts["stress"] = F.binary_cross_entropy_with_logits(
                predictions["stress"][mask], targets["stress"][mask]
            )
        else:
            parts["stress"] = predictions["stress"].sum() * 0.0
        total = total + float(lambdas.get("lambda_stress", 1.0)) * parts["stress"]

    if graph_regularizers is not None:
        mapping = {
            "pat": "lambda_pat",
            "ptt": "lambda_ptt",  # legacy v2 compatibility
            "direction": "lambda_dir",
            "sparsity": "lambda_sp",
        }
        for name, key in mapping.items():
            if name in graph_regularizers:
                parts[name] = graph_regularizers[name]
                total = total + float(lambdas.get(key, 0.0)) * parts[name]
    del edge_alpha, edge_beta  # unsupported evidential outputs are intentionally ignored
    parts["total"] = total
    return total, parts
