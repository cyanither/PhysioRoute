"""Forecasting heads for PI-CDGN and baselines."""

from __future__ import annotations

import torch
from torch import nn


class RegressionHead(nn.Module):
    """Simple linear regression head over pooled features."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        """Create a linear regression head."""

        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project pooled features to the requested target dimension."""

        return self.proj(x)


class MultiTaskForecastingHeads(nn.Module):
    """Shared multi-task output heads."""

    def __init__(self, d_model: int, short_steps: int, bp_steps: int, enabled: dict[str, bool] | None = None) -> None:
        """Initialize all configured heads."""

        super().__init__()
        enabled = enabled or {"waveform": True, "hr": True, "bp": True, "stress": True}
        self.enabled = enabled
        self.pool = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())
        if enabled.get("waveform", True):
            self.wave = nn.Linear(d_model, short_steps)
        if enabled.get("hr", True):
            self.hr = nn.Linear(d_model, 1)
        if enabled.get("bp", True):
            self.bp = nn.Linear(d_model, bp_steps * 2)
            self.bp_steps = int(bp_steps)
        if enabled.get("stress", True):
            self.stress = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Produce task predictions from ``(batch, nodes, time, dim)`` embeddings."""

        pooled = self.pool(h.mean(dim=(1, 2)))
        out: dict[str, torch.Tensor] = {}
        if hasattr(self, "wave"):
            out["wave"] = self.wave(pooled)
        if hasattr(self, "hr"):
            out["hr"] = self.hr(pooled)
        if hasattr(self, "bp"):
            out["bp"] = self.bp(pooled).reshape(h.shape[0], self.bp_steps, 2)
        if hasattr(self, "stress"):
            out["stress"] = self.stress(pooled)
        return out
