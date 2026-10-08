"""Physiology-Informed Causal Dynamic Graph Network."""

from __future__ import annotations

import torch
from torch import nn

from src.models.dyn_stgnn import DelayAwareGraphPropagation
from src.models.encoders import NodeTemporalEncoder
from src.models.graph_learner import GraphLearnerOutput, PhysiologyGraphLearner
from src.models.heads import MultiTaskForecastingHeads


class PICDGN(nn.Module):
    """PI-CDGN for multi-modal physiological forecasting."""

    def __init__(
        self,
        num_nodes: int,
        in_channels: int = 1,
        d_model: int = 128,
        target_fs: float = 64.0,
        chunk_sec: float = 2.0,
        short_steps: int = 640,
        bp_steps: int = 1920,
        message_passing_steps: int = 2,
        tau_min: float = 0.0,
        tau_max: float = 0.8,
        node_names: list[str] | None = None,
        head_enabled: dict[str, bool] | None = None,
        graph_type: str = "dynamic",
        directed: bool = True,
        delay_mode: str = "learned_pat_prior",
        delay_init: float | None = None,
        use_edge_uncertainty: bool = True,
        temporal_downsample: int = 8,
    ) -> None:
        """Create the PI-CDGN model."""

        super().__init__()
        temporal_downsample = max(1, int(temporal_downsample))
        encoded_fs = float(target_fs) / temporal_downsample
        chunk_size = max(1, int(round(chunk_sec * encoded_fs)))
        self.node_names = node_names
        self.encoder = NodeTemporalEncoder(
            in_channels=in_channels,
            d_model=d_model,
            downsample_factor=temporal_downsample,
        )
        self.graph_learner = PhysiologyGraphLearner(
            num_nodes=num_nodes,
            d_model=d_model,
            chunk_size=chunk_size,
            tau_min=tau_min,
            tau_max=tau_max,
            graph_type=graph_type,
            directed=directed,
            delay_mode=delay_mode,
            delay_init=delay_init,
            use_edge_uncertainty=use_edge_uncertainty,
        )
        self.propagation = DelayAwareGraphPropagation(
            d_model=d_model, num_steps=message_passing_steps, fs=encoded_fs
        )
        self.heads = MultiTaskForecastingHeads(
            d_model=d_model, short_steps=short_steps, bp_steps=bp_steps, enabled=head_enabled
        )

    def forward(
        self, x: torch.Tensor, current_pat: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor | GraphLearnerOutput]:
        """Run a forward pass.

        Args:
            x: Input tensor with shape ``(batch, nodes, time, features)``.

        Returns:
            Prediction dictionary plus graph tensors under ``graph``.
        """

        h = self.encoder(x)
        graph = self.graph_learner(h, delay_reference=current_pat)
        h = self.propagation(h, graph)
        predictions = self.heads(h)
        predictions["graph"] = graph
        pat_delay = self._extract_pat_delay(graph)
        if pat_delay is not None:
            predictions["pat_delay"] = pat_delay
        return predictions

    def _extract_pat_delay(self, graph: GraphLearnerOutput) -> torch.Tensor | None:
        """Return the learned ECG-to-PPG delay for PAT recovery metrics."""

        if not self.node_names or "ecg" not in self.node_names:
            return None
        src_idx = self.node_names.index("ecg")
        for dst_name in ("finger_ppg_ir", "wrist_bvp", "ppg"):
            if dst_name in self.node_names:
                dst_idx = self.node_names.index(dst_name)
                return graph.delay[..., dst_idx, src_idx].mean(dim=1, keepdim=True)
        return None
