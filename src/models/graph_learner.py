"""Pairwise physiology-informed causal graph learner."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GraphLearnerOutput:
    """Directed edge weights and transport delays for each temporal chunk."""

    edge_weight: torch.Tensor
    delay: torch.Tensor
    chunk_size: int


def gaussian_soft_delay(h: torch.Tensor, delays: torch.Tensor, fs: float, sigma: float = 1.0) -> torch.Tensor:
    """Apply a differentiable per-sample temporal delay with linear sampling."""

    del sigma
    if h.ndim != 3:
        raise ValueError(f"Expected h to have shape (batch, time, dim), got {tuple(h.shape)}")
    batch, time, _ = h.shape
    if delays.ndim == 1:
        delays = delays[:, None].expand(batch, time)
    if delays.shape != (batch, time):
        raise ValueError(f"Expected delays shape {(batch, time)}, got {tuple(delays.shape)}")
    if time == 1:
        return h
    base_idx = torch.arange(time, device=h.device, dtype=h.dtype)[None, :]
    source_pos = (base_idx - delays.to(device=h.device, dtype=h.dtype) * float(fs)).clamp(0, time - 1)
    left = torch.floor(source_pos).long()
    right = (left + 1).clamp_max(time - 1)
    fraction = (source_pos - left.to(source_pos.dtype)).unsqueeze(-1)
    gather_left = left.unsqueeze(-1).expand(batch, time, h.shape[-1])
    gather_right = right.unsqueeze(-1).expand(batch, time, h.shape[-1])
    left_value = torch.gather(h, dim=1, index=gather_left)
    right_value = torch.gather(h, dim=1, index=gather_right)
    return left_value * (1.0 - fraction) + right_value * fraction


def gumbel_sigmoid(logits: torch.Tensor, temperature: float, training: bool) -> torch.Tensor:
    """Differentiable Bernoulli relaxation used during training."""

    if not training:
        return torch.sigmoid(logits)
    uniform = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
    logistic_noise = torch.log(uniform) - torch.log1p(-uniform)
    return torch.sigmoid((logits + logistic_noise) / max(float(temperature), 1e-6))


class PhysiologyGraphLearner(nn.Module):
    """Infer directed edges from explicit target-source node pairs."""

    def __init__(
        self,
        num_nodes: int,
        d_model: int,
        chunk_size: int,
        tau_min: float = 0.0,
        tau_max: float = 0.5,
        hidden_dim: int | None = None,
        temperature: float = 1.0,
        graph_type: str = "dynamic",
        directed: bool = True,
        delay_mode: str = "learned_pat_prior",
        delay_init: float | None = None,
        use_edge_uncertainty: bool | None = None,
    ) -> None:
        super().__init__()
        del use_edge_uncertainty  # retained so legacy configs fail gracefully
        self.num_nodes = int(num_nodes)
        self.chunk_size = int(chunk_size)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.temperature = float(temperature)
        self.graph_type = str(graph_type)
        self.directed = bool(directed)
        self.delay_mode = str(delay_mode)
        if delay_init is None:
            delay_init = 0.5 * (self.tau_min + self.tau_max)
        if not self.tau_min < self.tau_max:
            raise ValueError("tau_min must be smaller than tau_max")
        if not self.tau_min <= float(delay_init) <= self.tau_max:
            raise ValueError(
                f"delay_init={delay_init} must lie in [{self.tau_min}, {self.tau_max}]"
            )
        init_fraction = (float(delay_init) - self.tau_min) / (self.tau_max - self.tau_min)
        init_fraction = min(max(init_fraction, 1e-6), 1.0 - 1e-6)
        delay_logit = math.log(init_fraction / (1.0 - init_fraction))
        hidden = hidden_dim or d_model * 2
        self.edge_head = nn.Sequential(
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 2),
        )
        self.global_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.num_nodes * self.num_nodes * 2),
        )
        self.static_raw = nn.Parameter(torch.zeros(self.num_nodes, self.num_nodes, 2))
        final_layer = self.edge_head[-1]
        if isinstance(final_layer, nn.Linear):
            with torch.no_grad():
                final_layer.weight[1].zero_()
                final_layer.bias[1].fill_(delay_logit)
        global_final = self.global_head[-1]
        if isinstance(global_final, nn.Linear):
            with torch.no_grad():
                global_final.weight[1::2].zero_()
                global_final.bias[1::2].fill_(delay_logit)
        with torch.no_grad():
            self.static_raw[..., 1].fill_(delay_logit)
        eye = torch.eye(self.num_nodes, dtype=torch.bool)
        self.register_buffer("self_mask", eye.view(1, 1, self.num_nodes, self.num_nodes), persistent=False)

    def set_temperature(self, temperature: float) -> None:
        self.temperature = float(temperature)

    @staticmethod
    def _pairwise(state: torch.Tensor) -> torch.Tensor:
        """Create [target, source, target-source, target*source] features."""

        target = state.unsqueeze(-2)
        source = state.unsqueeze(-3)
        target, source = torch.broadcast_tensors(target, source)
        return torch.cat((target, source, target - source, target * source), dim=-1)

    def forward(self, h: torch.Tensor, delay_reference: torch.Tensor | None = None) -> GraphLearnerOutput:
        if h.ndim != 4:
            raise ValueError(f"Expected h shape (batch, nodes, time, dim), got {tuple(h.shape)}")
        batch, nodes, time, dim = h.shape
        if nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {nodes}")
        chunks = (time + self.chunk_size - 1) // self.chunk_size
        pad = chunks * self.chunk_size - time
        h_state = F.pad(h, (0, 0, 0, pad), mode="replicate") if pad else h
        if self.graph_type == "static":
            raw = self.static_raw.view(1, 1, nodes, nodes, 2).expand(batch, chunks, nodes, nodes, 2)
        elif self.graph_type in {"learned_static", "static_learned"}:
            state = h.mean(dim=2)
            raw = self.edge_head(self._pairwise(state)).unsqueeze(1).expand(batch, chunks, nodes, nodes, 2)
        elif self.graph_type in {"global_dynamic", "global_pooled"}:
            state = h_state.reshape(batch, nodes, chunks, self.chunk_size, dim).mean(dim=(1, 3))
            raw = self.global_head(state).reshape(batch, chunks, nodes, nodes, 2)
        else:
            state = h_state.reshape(batch, nodes, chunks, self.chunk_size, dim).mean(dim=3).permute(0, 2, 1, 3)
            raw = self.edge_head(self._pairwise(state))
        edge_weight = gumbel_sigmoid(raw[..., 0], self.temperature, self.training)
        if delay_reference is not None:
            reference = delay_reference.to(device=h.device, dtype=h.dtype).reshape(batch, 1, 1, 1)
            max_delta = 0.25 * (self.tau_max - self.tau_min)
            delay = (reference + max_delta * torch.tanh(raw[..., 1])).clamp(self.tau_min, self.tau_max)
        else:
            delay = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(raw[..., 1])
        if self.delay_mode in {"zero", "none", "no_delay"}:
            delay = torch.zeros_like(delay)
        if not self.directed:
            edge_weight = 0.5 * (edge_weight + edge_weight.transpose(-1, -2))
            delay = 0.5 * (delay + delay.transpose(-1, -2))
        edge_weight = edge_weight.masked_fill(self.self_mask, 0.0)
        delay = delay.masked_fill(self.self_mask, 0.0)
        return GraphLearnerOutput(edge_weight=edge_weight, delay=delay, chunk_size=self.chunk_size)

    def regularization_losses(
        self,
        graph: GraphLearnerOutput,
        node_names: Sequence[str] | None = None,
        pat_reference: torch.Tensor | None = None,
        direction_priors: Mapping[tuple[str, str], float] | None = None,
        use_pat_prior: bool = True,
        use_direction_prior: bool = True,
        use_sparsity: bool = True,
        **legacy: object,
    ) -> dict[str, torch.Tensor]:
        """Compute only regularizers supported by observed targets or declared directions."""

        if pat_reference is None and "ptt_reference" in legacy:
            value = legacy["ptt_reference"]
            if isinstance(value, torch.Tensor):
                pat_reference = value
        if "use_ptt_prior" in legacy:
            use_pat_prior = bool(legacy["use_ptt_prior"])
        weights, delays = graph.edge_weight, graph.delay
        zero = weights.sum() * 0.0
        losses: dict[str, torch.Tensor] = {}
        if use_sparsity:
            losses["sparsity"] = weights.mean()
        if use_direction_prior:
            losses["direction"] = zero
        if node_names is None:
            return losses
        index = {name: idx for idx, name in enumerate(node_names)}
        priors = direction_priors or {
            ("ecg", "wrist_bvp"): 1.0,
            ("ecg", "finger_ppg_ir"): 1.0,
            ("acc_x", "finger_ppg_ir"): 1.0,
            ("wrist_acc_x", "wrist_bvp"): 1.0,
        }
        if use_direction_prior:
            terms = []
            for (src, dst), margin in priors.items():
                if src in index and dst in index:
                    forward = weights[..., index[dst], index[src]]
                    reverse = weights[..., index[src], index[dst]]
                    terms.append(F.relu(reverse - forward + float(margin) * 0.05).mean())
            if terms:
                losses["direction"] = torch.stack(terms).mean()
        if use_pat_prior and pat_reference is not None:
            for src, dst in (("ecg", "finger_ppg_ir"), ("ecg", "wrist_bvp"), ("ecg", "ppg")):
                if src in index and dst in index:
                    learned = delays[..., index[dst], index[src]].mean(dim=1)
                    ref = pat_reference.to(device=learned.device, dtype=learned.dtype).reshape(-1)
                    mask = torch.isfinite(ref)
                    losses["pat"] = F.l1_loss(learned[mask] * 1000.0, ref[mask] * 1000.0) if mask.any() else learned.sum() * 0.0
                    break
        return losses
