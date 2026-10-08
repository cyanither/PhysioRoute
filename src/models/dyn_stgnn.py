"""Dynamic delay-aware spatio-temporal graph propagation."""

from __future__ import annotations

import torch
from torch import nn

from src.models.graph_learner import GraphLearnerOutput, gaussian_soft_delay


class DelayAwareGraphPropagation(nn.Module):
    """Delay-aware directed message passing with GRU updates."""

    def __init__(self, d_model: int, num_steps: int = 2, fs: float = 64.0, delay_sigma: float = 1.0) -> None:
        """Initialize propagation."""

        super().__init__()
        self.num_steps = int(num_steps)
        self.fs = float(fs)
        self.delay_sigma = float(delay_sigma)
        self.message_mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.update = nn.GRUCell(d_model, d_model)

    def _expand_chunks(self, values: torch.Tensor, time: int, chunk_size: int) -> torch.Tensor:
        expanded = values.repeat_interleave(chunk_size, dim=1)
        return expanded[:, :time]

    def forward(self, h: torch.Tensor, graph: GraphLearnerOutput) -> torch.Tensor:
        """Propagate node embeddings.

        Args:
            h: Tensor with shape ``(batch, nodes, time, d_model)``.
            graph: Chunked dynamic graph output.

        Returns:
            Updated embeddings with shape ``(batch, nodes, time, d_model)``.
        """

        if h.ndim != 4:
            raise ValueError(f"Expected h shape (batch, nodes, time, dim), got {tuple(h.shape)}")
        batch, nodes, time, dim = h.shape
        weights = self._expand_chunks(graph.edge_weight, time, graph.chunk_size)
        delays = self._expand_chunks(graph.delay, time, graph.chunk_size)
        out = h
        for _ in range(self.num_steps):
            messages = torch.zeros_like(out)
            for dst in range(nodes):
                agg = torch.zeros(batch, time, dim, device=h.device, dtype=h.dtype)
                for src in range(nodes):
                    if src == dst:
                        continue
                    delayed = gaussian_soft_delay(
                        out[:, src], delays[:, :, dst, src], fs=self.fs, sigma=self.delay_sigma
                    )
                    msg = self.message_mlp(delayed) * weights[:, :, dst, src].unsqueeze(-1)
                    agg = agg + msg
                messages[:, dst] = agg
            updated = self.update(messages.reshape(-1, dim), out.reshape(-1, dim))
            out = updated.reshape(batch, nodes, time, dim)
        return out


class IdentityDynamicGraphPropagation(nn.Module):
    """Placeholder propagation module."""

    def forward(self, h: torch.Tensor, *_: torch.Tensor) -> torch.Tensor:
        """Return node embeddings unchanged."""

        return h
