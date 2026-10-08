#!/usr/bin/env python3
"""Build a lightweight visual-QA contact sheet using the plotting backend."""

from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
FIGDIR = ROOT / "final/PI_DDGN__Physiology_Informed_Causal_Delay_Graph_Learning_for_Multimodal_Wearable_Health_Forecasting/figures_v3"
STEMS = [
    "fig_v3_architecture",
    "fig_v3_main_hr",
    "fig_v3_ablation_pat",
    "fig_v3_pat_diagnostics",
    "fig_v3_robustness",
]

fig, axes = plt.subplots(3, 2, figsize=(8.0, 7.0))
for ax, stem in zip(axes.flat, STEMS):
    ax.imshow(plt.imread(FIGDIR / f"{stem}.png"))
    ax.set_title(stem.replace("fig_v3_", "").replace("_", " "), fontsize=9)
    ax.axis("off")
axes.flat[-1].axis("off")
fig.tight_layout(pad=0.6)
fig.savefig(FIGDIR / "qa_contact_sheet.jpg", dpi=60, pil_kwargs={"quality": 42, "optimize": True})
plt.close(fig)
