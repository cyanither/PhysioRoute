#!/usr/bin/env python3
"""Generate submission figures for the audited v3 manuscript.

Every quantitative panel is derived from results_v3 JSON or COMPLETE-run
prediction artifacts.  The script does not read results_v2.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "final/PI_DDGN__Physiology_Informed_Causal_Delay_Graph_Learning_for_Multimodal_Wearable_Health_Forecasting"
OUT = PAPER / "figures_v3"
SOURCE = OUT / "source_data"
OUT.mkdir(parents=True, exist_ok=True)
SOURCE.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.0,
        "axes.labelsize": 7.5,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

BLUE = "#4C78A8"
ORANGE = "#E17C36"
GREEN = "#59A14F"
RED = "#C44E52"
PURPLE = "#8172B3"
TEAL = "#4C9F9B"
GRAY = "#9A9A9A"
LIGHT = "#E8E8E8"
INK = "#252525"


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def aggregate_pat(relative: str) -> tuple[float, float]:
    values = [
        load_json(f"{relative}/seed{seed}/test_metrics.json")["pat_mae_ms"]
        for seed in range(42, 47)
    ]
    return float(np.mean(values)), float(np.std(values, ddof=1))


def save_figure(fig: plt.Figure, stem: str) -> None:
    for suffix, kwargs in (
        ("pdf", {}),
        ("svg", {}),
        ("png", {"dpi": 300}),
        ("tiff", {"dpi": 600}),
    ):
        fig.savefig(OUT / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.07, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="top")


def write_csv(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with (SOURCE / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def box(ax: plt.Axes, xy: tuple[float, float], width: float, height: float, text: str, color: str) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        linewidth=0.8,
        edgecolor=color,
        facecolor=mpl.colors.to_rgba(color, 0.12),
    )
    ax.add_patch(patch)
    ax.text(xy[0] + width / 2, xy[1] + height / 2, text, ha="center", va="center", color=INK, fontsize=7)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], color: str = GRAY) -> None:
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=8, linewidth=0.9, color=color))


def architecture_figure() -> None:
    fig, ax = plt.subplots(figsize=(7.16, 3.35))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    modalities = [("ECG", 0.80), ("PPG", 0.64), ("Resp./EDA", 0.48), ("ACC/Temp.", 0.32)]
    for label, y in modalities:
        box(ax, (0.02, y), 0.10, 0.10, label, BLUE)
        arrow(ax, (0.12, y + 0.05), (0.18, y + 0.05), BLUE)
    ax.text(0.07, 0.94, "60-s multimodal input", ha="center", fontsize=7.5, fontweight="bold")

    box(ax, (0.18, 0.30), 0.15, 0.62, "Shared node-wise encoder\n\nCausal dilated TCN\n+ causal Transformer\n\nNo cross-node mixing", TEAL)
    arrow(ax, (0.33, 0.61), (0.39, 0.61), TEAL)
    box(ax, (0.39, 0.52), 0.19, 0.24, "Ordered pair features\n$[s_i,s_j,s_i-s_j,s_i\\odot s_j]$\n\nEdge weight $a_{ij,k}$\nLatent delay $\\delta_{ij,k}$", ORANGE)
    ax.text(0.485, 0.80, "2-s chunks", ha="center", fontsize=6.5, color=ORANGE)
    arrow(ax, (0.58, 0.64), (0.64, 0.64), ORANGE)
    box(ax, (0.64, 0.48), 0.15, 0.32, "Continuous delayed\nsource sampling\n\nDirected message\npropagation", PURPLE)
    arrow(ax, (0.79, 0.64), (0.85, 0.64), PURPLE)
    box(ax, (0.85, 0.57), 0.13, 0.14, "Future mean HR\n8-s primary", GREEN)

    box(ax, (0.39, 0.15), 0.19, 0.18, "Training constraints\nHR MAE + PAT MAE*\n+ direction + sparsity", RED)
    arrow(ax, (0.485, 0.33), (0.485, 0.52), RED)
    ax.text(0.60, 0.17, "*PTT-PPG only; future ECG-to-finger-IR PAT", fontsize=6.2, color=RED, va="center")

    box(ax, (0.69, 0.12), 0.29, 0.20, "Leakage-resistant evaluation\nCausal preprocessing  |  input-only z-score\nsubject-disjoint splits  |  five seeds\npersistence, ablations, negative controls", BLUE)
    ax.text(0.50, 0.03, "Input-period PAT is external baseline metadata and is never supplied to PI-DDGN", ha="center", fontsize=6.5, color=INK)
    save_figure(fig, "fig_v3_architecture")


def main_hr_figure() -> None:
    summary = load_json("results_v3/revised_primary_summary.json")["groups"]
    simple = load_json("results_v3/simple_baselines.json")
    datasets = [("wesad", "WESAD"), ("ppg_dalia", "PPG-DaLiA"), ("ptt_ppg", "PTT-PPG")]
    models = ["deep_ppg", "gwn", "itransformer", "lstm", "mtgnn", "patchtst", "picdgn_v3"]
    labels = ["Deep PPG", "Graph WaveNet", "iTransformer", "LSTM", "MTGNN", "PatchTST", "PhysioRoute"]
    baseline_color = "#677C8D"
    accent_color = "#BF414B"
    colors = [baseline_color] * 6 + [accent_color]
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.80), sharey=False)
    for panel, ax, (key, title) in zip("abc", axes, datasets):
        persistence = float(simple[title]["current_hr_persistence"]["window_mae"])
        interval_bounds = [0.0]
        ax.axhspan(5.55, 6.45, color=accent_color, alpha=0.07, linewidth=0, zorder=0)
        for y in range(len(labels) - 1):
            ax.axhline(y, color="#EDF0F3", linewidth=0.55, zorder=0)
        for y, (model, label, color) in enumerate(zip(models, labels, colors)):
            record = summary[f"{key}/{model}"]
            mean = float(record["window_hr_mae_mean"])
            sd = float(record["window_hr_mae_sd"])
            delta = mean - persistence
            interval_bounds.extend((delta - sd, delta + sd))
            proposed = model == "picdgn_v3"
            ax.errorbar(
                delta, y, xerr=sd, fmt="D" if proposed else "o",
                markersize=5.0 if proposed else 4.0,
                color=color, ecolor=color, markeredgecolor="white", markeredgewidth=0.55,
                capsize=2.5, capthick=1.0, elinewidth=1.25 if proposed else 1.0, zorder=3,
            )
            rows.append({"dataset": title, "model": label, "mean_window_mae_bpm": mean,
                         "seed_sd_bpm": sd, "persistence_mae_bpm": persistence, "delta_vs_persistence_bpm": delta})
        ax.axvline(0, color="#7D8993", linewidth=0.9, linestyle=(0, (3, 3)), zorder=2)
        lower, upper = min(interval_bounds), max(interval_bounds)
        padding = max((upper - lower) * 0.09, 0.005)
        default_limits = (-0.03, 0.03) if key == "ptt_ppg" else (-0.55, 0.15)
        ax.set_xlim(min(default_limits[0], lower - padding), max(default_limits[1], upper + padding))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
        ax.set_title(f"{panel}   {title}", loc="left", fontweight="bold", fontsize=8.5, color=INK, pad=12)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if ax is axes[0] else [])
        ax.set_ylim(6.65, -0.65)
        ax.tick_params(axis="y", length=0, pad=8, labelsize=7.2, labelcolor=INK)
        ax.tick_params(axis="x", length=3, width=0.6, color="#B4BDC5", labelcolor="#53616E", pad=4)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#B4BDC5")
        ax.spines["bottom"].set_linewidth(0.6)
        ax.grid(axis="x", color="#EDF0F3", linewidth=0.5, zorder=0)
    axes[0].get_yticklabels()[-1].set_color(accent_color)
    axes[0].get_yticklabels()[-1].set_fontweight("bold")
    fig.text(0.585, 0.125, "$\\Delta$MAE vs persistence (bpm)  ·  Lower is better",
             ha="center", fontsize=7.5, color=INK)
    fig.text(0.585, 0.035, "Mean ± seed SD   ·   Dashed line: persistence   ·   Panel scales differ",
             ha="center", fontsize=6.3, color="#687580")
    fig.subplots_adjust(left=0.18, right=0.99, bottom=0.28, top=0.87, wspace=0.23)
    write_csv("fig_v3_main_hr.csv", rows)
    save_figure(fig, "fig_v3_main_hr")


def ablation_pat_figure() -> None:
    supplemental = load_json("results_v3/supplemental_summary.json")
    main = load_json("results_v3/revised_primary_summary.json")["groups"]
    original = load_json("results_v3/canonical_h8_summary.json")["groups"]
    pat_base = load_json("results_v3/pat_signal_baselines.json")
    datasets = [("wesad", "WESAD"), ("ppg_dalia", "PPG-DaLiA"), ("ptt_ppg", "PTT-PPG")]
    variants = ["proposed", "chunk_dynamic", "global_pooled", "no_delay", "undirected"]
    labels = ["Proposed", "Chunk dynamic", "Global pooled", "No delay", "Undirected"]
    baseline_color = "#677C8D"
    accent_color = "#BF414B"
    colors = [accent_color] + [baseline_color] * 4
    rows: list[dict[str, object]] = []
    fig = plt.figure(figsize=(7.16, 2.85))
    # The empty grid column reserves space for the independent PAT row labels.
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 0.48, 1.22])
    axes = [fig.add_subplot(grid[0, i]) for i in (0, 1, 2, 4)]
    y = np.arange(len(variants))
    for panel_index, (ax, (dataset, title)) in enumerate(zip(axes[:3], datasets)):
        values = []
        errors = []
        for variant, label in zip(variants, labels):
            if variant == "proposed":
                record = main[f"{dataset}/picdgn_v3"]
                mean, sd = record["window_hr_mae_mean"], record["window_hr_mae_sd"]
            elif variant == "chunk_dynamic":
                record = original[f"{dataset}/picdgn_v3"]
                mean, sd = record["window_hr_mae_mean"], record["window_hr_mae_sd"]
            else:
                record = supplemental[f"ablation/{dataset}/{variant}"]
                mean, sd = record["hr_mae_mean"], record["hr_mae_sd"]
            values.append(float(mean)); errors.append(float(sd))
            rows.append({"panel": title, "variant": label, "mean": mean, "sd": sd, "unit": "bpm"})
        ax.axhspan(-0.42, 0.42, color=accent_color, alpha=0.07, linewidth=0, zorder=0)
        for yi, value, error, color in zip(y, values, errors, colors):
            ax.axhline(yi, color="#EDF0F3", linewidth=0.55, zorder=0)
            ax.errorbar(value, yi, xerr=error, fmt="D" if yi == 0 else "o", color=color, ecolor=color,
                        markersize=4.8 if yi == 0 else 3.8, markeredgecolor="white", markeredgewidth=0.5,
                        capsize=2.3, capthick=0.9, elinewidth=1.2 if yi == 0 else 1.0, zorder=3)
        lower = min(np.asarray(values) - np.asarray(errors))
        upper = max(np.asarray(values) + np.asarray(errors))
        pad = max((upper - lower) * 0.18, 0.012)
        ax.set_xlim(lower - pad, upper + pad)
        ax.set_yticks(y)
        ax.set_yticklabels(["PhysioRoute", *labels[1:]] if panel_index == 0 else [])
        ax.set_ylim(4.55, -0.55)
        ax.set_title(f"{'abc'[panel_index]}   {title}", loc="left", fontweight="bold", fontsize=8, pad=12, color=INK)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, steps=[1, 2, 2.5, 5, 10]))
    axes[0].get_yticklabels()[0].set_color(accent_color)
    axes[0].get_yticklabels()[0].set_fontweight("bold")

    tuned = aggregate_pat("results_v3/pat_tuning/validation/tau250_init160_lam001")
    pat_only = aggregate_pat("results_v3/pat_tuning/controls/pat_only")
    shuffled = aggregate_pat("results_v3/pat_tuning/controls/pat_shuffled")
    reversed_time = aggregate_pat("results_v3/pat_tuning/controls/time_reversed")
    scrambled = aggregate_pat("results_v3/pat_tuning/controls/channels_scrambled")
    pat_items = [
        ("Persistence", pat_base["current_pat_persistence"]["window_mae_ms"], 0.0, BLUE),
        ("PAT-tuned", tuned[0], tuned[1], RED),
        ("PAT only", pat_only[0], pat_only[1], ORANGE),
        ("Shuffled", shuffled[0], shuffled[1], GRAY),
        ("Reversed", reversed_time[0], reversed_time[1], PURPLE),
        ("Scrambled", scrambled[0], scrambled[1], TEAL),
    ]
    ax = axes[3]
    y = np.arange(len(pat_items))
    ax.barh(y, [item[1] for item in pat_items], color=[baseline_color, accent_color, *(["#A6B3BE"] * 4)],
            height=0.54, zorder=3)
    # Deterministic persistence has no seed uncertainty, so do not draw a zero-length cap.
    ax.errorbar([item[1] for item in pat_items[1:]], y[1:], xerr=[item[2] for item in pat_items[1:]],
                fmt="none", ecolor="#43515D", capsize=2.2, capthick=0.8, elinewidth=0.85, zorder=4)
    ax.set_yticks(y); ax.set_yticklabels([item[0] for item in pat_items]); ax.invert_yaxis()
    ax.set_ylim(5.6, -0.6)
    ax.set_xlim(0, max(item[1] + item[2] for item in pat_items) * 1.12)
    ax.set_xticks([0, 20, 40])
    ax.set_xlabel("PAT MAE (ms)", labelpad=7)
    ax.set_title("d   PAT controls", loc="left", fontweight="bold", fontsize=8, pad=12, color=INK)
    ax.get_yticklabels()[1].set_color(accent_color)
    ax.get_yticklabels()[1].set_fontweight("bold")
    for label, mean, sd, _ in pat_items:
        rows.append({"panel": "PAT controls", "variant": label, "mean": mean, "sd": sd, "unit": "ms"})
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="x", color="#EDF0F3", linewidth=0.55)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#B4BDC5")
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(axis="y", length=0, pad=7, labelsize=6.7)
        ax.tick_params(axis="x", length=3, width=0.6, color="#B4BDC5", labelcolor="#53616E", pad=4)
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.27, top=0.86, wspace=0.26)
    hr_center = (axes[0].get_position().x0 + axes[2].get_position().x1) / 2
    fig.text(hr_center, 0.135, "HR MAE (bpm)", ha="center", fontsize=7.5, color=INK)
    fig.text(0.56, 0.035, "Mean ± seed SD   ·   Lower is better   ·   Panel scales differ",
             ha="center", fontsize=6.3, color="#687580")
    write_csv("fig_v3_ablation_pat.csv", rows)
    save_figure(fig, "fig_v3_ablation_pat")


def pat_diagnostics_figure() -> None:
    root = ROOT / "results_v3/pat_tuning/validation/tau250_init160_lam001"
    predictions = []
    target = subject = None
    for path in sorted(root.glob("seed*/predictions.npz")):
        with np.load(path) as archive:
            predictions.append(np.asarray(archive["pred_pat_delay"], dtype=np.float64).reshape(-1) * 1000)
            if target is None:
                target = np.asarray(archive["target_pat"], dtype=np.float64).reshape(-1) * 1000
                subject = np.asarray(archive["meta_subject"]).astype(str).reshape(-1)
    assert target is not None and subject is not None and len(predictions) == 5
    ensemble = np.mean(predictions, axis=0)
    persistence = np.load(ROOT / "data/processed_v3/ptt_ppg_h8/test/baseline_current_pat.npy").astype(np.float64) * 1000
    subjects = sorted(np.unique(subject).tolist())
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.95), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    limits = (75, 325)
    baseline_color = "#677C8D"
    accent_color = "#BF414B"
    density_maps = [
        mpl.colors.LinearSegmentedColormap.from_list("pat_persistence", ["#D5E1EB", "#6E92AD", "#244B68"]),
        mpl.colors.LinearSegmentedColormap.from_list("pat_physioroute", ["#F2D7D9", "#D8777E", "#972F39"]),
    ]
    densities = []
    for ax, values, cmap in (
        (axes[0], persistence, density_maps[0]),
        (axes[1], ensemble, density_maps[1]),
    ):
        densities.append(ax.hexbin(target, values, gridsize=28, mincnt=1, cmap=cmap,
                                   linewidths=0, zorder=2))
        ax.plot(limits, limits, color="#78848D", linewidth=0.9, linestyle=(0, (4, 3)), zorder=3)
        mae = float(np.mean(np.abs(values - target)))
        corr = float(np.corrcoef(values, target)[0, 1])
        ax.text(0.04, 0.96, f"MAE = {mae:.2f} ms\n$r$ = {corr:.2f}", transform=ax.transAxes,
                va="top", fontsize=7, color=INK, linespacing=1.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.92, "pad": 2.5}, zorder=4)
        ax.set_xlim(limits); ax.set_ylim(limits); ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([100, 200, 300]); ax.set_yticks([100, 200, 300])
        ax.set_xlabel("Future PAT target (ms)"); ax.set_ylabel("Prediction (ms)")
    # Use the same logarithmic count scale in both density panels.  The
    # non-white low end keeps sparsely occupied bins visible as well.
    density_norm = mpl.colors.LogNorm(vmin=1, vmax=max(float(h.get_array().max()) for h in densities))
    for density in densities:
        density.set_norm(density_norm)
    x = np.arange(len(subjects)); width = 0.32
    persist_subject = [float(np.mean(np.abs(persistence[subject == s] - target[subject == s]))) for s in subjects]
    model_subject = [float(np.mean(np.abs(ensemble[subject == s] - target[subject == s]))) for s in subjects]
    axes[2].bar(x - width / 2, persist_subject, width, color=baseline_color, label="Persistence", zorder=3)
    axes[2].bar(x + width / 2, model_subject, width, color=accent_color, label="PhysioRoute", zorder=3)
    axes[2].set_xticks(x); axes[2].set_xticklabels(subjects)
    axes[2].set_xlabel("Held-out subject")
    axes[2].set_ylabel("Subject PAT MAE (ms)")
    axes[2].set_ylim(0, 60)
    axes[2].set_box_aspect(1 / 1.15)
    axes[2].set_yticks([0, 20, 40, 60])
    handles, legend_labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(0.54, 1.0),
               ncol=2, fontsize=7, handlelength=1.6, handleheight=0.9, columnspacing=2.5)
    for label, ax in zip("abc", axes):
        ax.set_axisbelow(True)
        ax.grid(axis="y" if ax is axes[2] else "both", color="#EDF0F3", linewidth=0.55)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#B4BDC5")
            ax.spines[side].set_linewidth(0.6)
        ax.tick_params(length=3, width=0.6, color="#B4BDC5", labelcolor="#53616E", pad=4)
        ax.xaxis.label.set_color(INK); ax.yaxis.label.set_color(INK)
        ax.text(0, 1.045, label, transform=ax.transAxes, fontsize=8.5, fontweight="bold", color=INK)
    for i, current in enumerate(target):
        rows.append({"window": i, "subject": subject[i], "target_pat_ms": current,
                     "persistence_pat_ms": persistence[i], "pi_ddgn_ensemble_pat_ms": ensemble[i]})
    fig.text(0.53, 0.035, "Darker hexagons: more windows (shared log-count scale)   ·   Dashed lines: ideal prediction",
             ha="center", fontsize=6.2, color="#687580")
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.25, top=0.82, wspace=0.48)
    write_csv("fig_v3_pat_diagnostics.csv", rows)
    save_figure(fig, "fig_v3_pat_diagnostics")


def robustness_figure() -> None:
    supplemental = load_json("results_v3/supplemental_summary.json")
    activity = load_json("results_v3/activity_strict_baselines.json")
    horizon = load_json("results_v3/horizon_target_audit.json")
    rows: list[dict[str, object]] = []
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.70))
    baseline_color = "#677C8D"
    accent_color = "#BF414B"
    error_style = {"elinewidth": 0.9, "capthick": 0.9, "ecolor": "#43515D"}

    modality = [("ECG", "ecg_only"), ("PPG", "ppg_only"), ("PPG+ACC", "ppg_acc"), ("All", "all_modalities")]
    values = [supplemental[f"modality/ppg_dalia/{key}/patchtst"]["hr_mae_mean"] for _, key in modality]
    errors = [supplemental[f"modality/ppg_dalia/{key}/patchtst"]["hr_mae_sd"] for _, key in modality]
    axes[0, 0].bar(range(4), values, yerr=errors, color=["#A6B3BE", baseline_color, "#739D9A", accent_color],
                   width=0.58, capsize=2.5, error_kw=error_style, zorder=3)
    axes[0, 0].set_ylim(0, max(np.asarray(values) + np.asarray(errors)) * 1.16)
    axes[0, 0].set_xticks(range(4)); axes[0, 0].set_xticklabels([x[0] for x in modality])
    axes[0, 0].set_ylabel("HR MAE (bpm)"); axes[0, 0].set_title("Modality test (no HR residual)", fontweight="bold")
    for (label, _), value, error in zip(modality, values, errors):
        rows.append({"panel": "modality", "condition": label, "mean": value, "sd": error, "unit": "bpm"})

    strict = [("Persistence", activity["current_hr_persistence"]["window_mae"], 0, BLUE),
              ("Linear HR", activity["linear_ar_current_hr"]["window_mae"], 0, GRAY),
              ("PatchTST", supplemental["activity_strict/ppg_dalia/patchtst"]["hr_mae_mean"], supplemental["activity_strict/ppg_dalia/patchtst"]["hr_mae_sd"], RED)]
    axes[0, 1].bar(range(3), [x[1] for x in strict], color=[baseline_color, "#A6B3BE", accent_color],
                   width=0.55, zorder=3)
    axes[0, 1].errorbar([2], [strict[2][1]], yerr=[strict[2][2]], fmt="none",
                        capsize=2.5, capthick=0.9, elinewidth=0.9, ecolor="#43515D", zorder=4)
    axes[0, 1].set_ylim(0, max(item[1] + item[2] for item in strict) * 1.17)
    axes[0, 1].set_xticks(range(3)); axes[0, 1].set_xticklabels([x[0] for x in strict])
    axes[0, 1].set_ylabel("HR MAE (bpm)"); axes[0, 1].set_title("Held-out activity", fontweight="bold")
    for label, mean, sd, _ in strict:
        rows.append({"panel": "activity", "condition": label, "mean": mean, "sd": sd, "unit": "bpm"})

    names = ["WESAD", "PPG-DaLiA", "PTT-PPG"]
    x = np.arange(3)
    p8 = [horizon[name]["8"]["test"]["current_hr_persistence_window_mae_bpm"] for name in names]
    p30 = [horizon[name]["30"]["test"]["current_hr_persistence_window_mae_bpm"] for name in names]
    m30 = [supplemental[f"horizon/h30/{key}/patchtst"]["hr_mae_mean"] for key in ("wesad", "ppg_dalia", "ptt_ppg")]
    axes[1, 0].plot(x, p8, "o--", color="#A6B3BE", markersize=4, linewidth=1.0, label="8-s persistence")
    axes[1, 0].plot(x, p30, "s--", color=baseline_color, markersize=6, markerfacecolor="white",
                    markeredgewidth=1.0, linewidth=1.0, label="30-s persistence", zorder=3)
    axes[1, 0].plot(x, m30, "D-", color=accent_color, markersize=3.5,
                    linewidth=1.0, label="30-s PatchTST", zorder=4)
    axes[1, 0].margins(x=0.10, y=0.13)
    axes[1, 0].set_xticks(x); axes[1, 0].set_xticklabels(names)
    axes[1, 0].set_ylabel("HR MAE (bpm)"); axes[1, 0].set_title("Horizon sensitivity", fontweight="bold")
    axes[1, 0].legend(ncol=1, loc="lower left", fontsize=6.3, handlelength=2, labelspacing=0.5,
                      frameon=True, facecolor="white", edgecolor="none", framealpha=0.94)
    for name, a, b, c in zip(names, p8, p30, m30):
        for condition, value in (("8-s persistence", a), ("30-s persistence", b), ("30-s PatchTST", c)):
            rows.append({"panel": "horizon", "condition": f"{name}: {condition}", "mean": value, "sd": 0, "unit": "bpm"})

    transfer = [("to WESAD", "cross/to_wesad/patchtst"), ("to DaLiA", "cross/to_dalia/patchtst"), ("to PTT", "cross/to_ptt_ppg/patchtst")]
    values = [supplemental[key]["hr_mae_mean"] for _, key in transfer]
    errors = [supplemental[key]["hr_mae_sd"] for _, key in transfer]
    axes[1, 1].bar(range(3), values, yerr=errors, color=baseline_color, width=0.55,
                   capsize=2.5, error_kw=error_style, zorder=3)
    axes[1, 1].set_ylim(0, max(np.asarray(values) + np.asarray(errors)) * 1.16)
    axes[1, 1].set_xticks(range(3)); axes[1, 1].set_xticklabels([x[0] for x in transfer])
    axes[1, 1].set_ylabel("Zero-shot HR MAE (bpm)"); axes[1, 1].set_title("Cross-dataset transfer", fontweight="bold")
    for (label, _), value, error in zip(transfer, values, errors):
        rows.append({"panel": "transfer", "condition": label, "mean": value, "sd": error, "unit": "bpm"})

    titles = ["Modality test (no HR residual)", "Held-out activity", "Horizon sensitivity", "Cross-dataset transfer"]
    for label, title, ax in zip("abcd", titles, axes.flat):
        ax.set_title("")
        ax.set_title(f"{label}   {title}", loc="left", fontweight="bold", fontsize=8, color=INK, pad=11)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#EDF0F3", linewidth=0.55)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color("#B4BDC5")
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(axis="y", length=0, labelcolor="#53616E", pad=5)
        ax.tick_params(axis="x", length=3, width=0.6, color="#B4BDC5", labelcolor=INK, pad=5)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.label.set_color(INK)
    fig.text(0.54, 0.025, "Lower is better   ·   Error bars: seed SD   ·   Panel scales differ",
             ha="center", fontsize=6.3, color="#687580")
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.13, top=0.93, hspace=0.56, wspace=0.30)
    write_csv("fig_v3_robustness.csv", rows)
    save_figure(fig, "fig_v3_robustness")


def main() -> None:
    architecture_figure()
    main_hr_figure()
    ablation_pat_figure()
    pat_diagnostics_figure()
    robustness_figure()
    manifest = {
        "backend": "Python/matplotlib",
        "source": "Audited results_v3 JSON and COMPLETE-run predictions only",
        "figures": sorted(path.name for path in OUT.glob("fig_v3_*.pdf")),
        "source_data": sorted(path.name for path in SOURCE.glob("*.csv")),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
