"""Visualization helpers for learned graphs and clinical metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def save_adjacency_heatmap(adjacency: np.ndarray, node_names: list[str], output_path: str | Path, title: str = "Learned adjacency") -> None:
    """Save an adjacency heatmap as PNG/PDF depending on path suffix."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(max(5, len(node_names) * 0.6), max(4, len(node_names) * 0.5)))
    sns.heatmap(adjacency, xticklabels=node_names, yticklabels=node_names, cmap="viridis", vmin=0.0)
    plt.title(title)
    plt.xlabel("Source")
    plt.ylabel("Target")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_delay_distribution(delays: dict[str, np.ndarray], output_path: str | Path, ptt_range: tuple[float, float] = (0.1, 0.4)) -> None:
    """Save per-edge delay histograms with the reference PTT band."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    for edge, values in delays.items():
        values = np.asarray(values).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size:
            sns.kdeplot(values * 1000.0, label=edge)
    plt.axvspan(ptt_range[0] * 1000.0, ptt_range[1] * 1000.0, color="tab:green", alpha=0.15, label="PTT prior")
    plt.xlabel("Delay (ms)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_bp_scatter(pred: np.ndarray, target: np.ndarray, output_path: str | Path) -> None:
    """Save SBP/DBP prediction scatter plots."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for idx, name in enumerate(("SBP", "DBP")):
        p = pred[..., idx].reshape(-1)
        t = target[..., idx].reshape(-1)
        mask = np.isfinite(t)
        axes[idx].scatter(t[mask], p[mask], s=4, alpha=0.35)
        lo = float(np.nanmin(t[mask])) if mask.any() else 0.0
        hi = float(np.nanmax(t[mask])) if mask.any() else 1.0
        axes[idx].plot([lo, hi], [lo, hi], color="black", linewidth=1)
        axes[idx].set_title(name)
        axes[idx].set_xlabel("Reference mmHg")
        axes[idx].set_ylabel("Predicted mmHg")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def save_bland_altman(pred: np.ndarray, target: np.ndarray, output_path: str | Path) -> None:
    """Save Bland-Altman plots for SBP and DBP."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for idx, name in enumerate(("SBP", "DBP")):
        p = pred[..., idx].reshape(-1)
        t = target[..., idx].reshape(-1)
        mask = np.isfinite(t)
        mean = (p[mask] + t[mask]) / 2.0
        diff = p[mask] - t[mask]
        axes[idx].scatter(mean, diff, s=4, alpha=0.35)
        bias = np.mean(diff) if diff.size else np.nan
        sd = np.std(diff) if diff.size else np.nan
        axes[idx].axhline(bias, color="black", linewidth=1)
        axes[idx].axhline(bias + 1.96 * sd, color="tab:red", linestyle="--", linewidth=1)
        axes[idx].axhline(bias - 1.96 * sd, color="tab:red", linestyle="--", linewidth=1)
        axes[idx].set_title(name)
        axes[idx].set_xlabel("Mean mmHg")
        axes[idx].set_ylabel("Prediction - reference")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def save_selective_prediction_curve(error: np.ndarray, uncertainty: np.ndarray, output_path: str | Path) -> None:
    """Save error versus retained coverage using uncertainty ranking."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error = np.asarray(error).reshape(-1)
    uncertainty = np.asarray(uncertainty).reshape(-1)
    mask = np.isfinite(error) & np.isfinite(uncertainty)
    order = np.argsort(uncertainty[mask])
    sorted_error = error[mask][order]
    coverages = np.linspace(0.05, 1.0, 20)
    risks = []
    for coverage in coverages:
        keep = max(1, int(round(coverage * sorted_error.size)))
        risks.append(float(np.mean(sorted_error[:keep])))
    plt.figure(figsize=(5, 4))
    plt.plot(coverages, risks, marker="o")
    plt.xlabel("Coverage")
    plt.ylabel("Mean retained error")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
