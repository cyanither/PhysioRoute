"""Shared baseline components with the PI-CDGN prediction interface."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from src.models.heads import MultiTaskForecastingHeads


class BaselineBase(nn.Module):
    """Base class exposing shared multi-task heads."""

    def __init__(self, d_model: int, short_steps: int, bp_steps: int) -> None:
        """Initialize shared heads."""

        super().__init__()
        self.heads = MultiTaskForecastingHeads(d_model=d_model, short_steps=short_steps, bp_steps=bp_steps)

    def _head_from_sequence(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """Apply heads to sequence embeddings with shape ``(batch, time, dim)``."""

        return self.heads(h[:, None])


def flatten_nodes(x: torch.Tensor) -> torch.Tensor:
    """Flatten node and feature dimensions into variates, returning ``(batch, time, vars)``."""

    if x.ndim != 4:
        raise ValueError(f"Expected x shape (batch, nodes, time, features), got {tuple(x.shape)}")
    batch, nodes, time, features = x.shape
    return x.permute(0, 2, 1, 3).reshape(batch, time, nodes * features)


class LSTMBaseline(BaselineBase):
    """Two-layer bidirectional LSTM baseline."""

    def __init__(self, num_nodes: int, in_channels: int = 1, hidden_size: int = 128, short_steps: int = 640, bp_steps: int = 1920) -> None:
        super().__init__(hidden_size * 2, short_steps, bp_steps)
        self.lstm = nn.LSTM(num_nodes * in_channels, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        seq, _ = self.lstm(flatten_nodes(x))
        return self._head_from_sequence(seq)


class TransformerBaseline(BaselineBase):
    """Vanilla temporal Transformer encoder baseline."""

    def __init__(self, num_nodes: int, in_channels: int = 1, d_model: int = 128, num_layers: int = 4, num_heads: int = 4, short_steps: int = 640, bp_steps: int = 1920) -> None:
        super().__init__(d_model, short_steps, bp_steps)
        self.proj = nn.Linear(num_nodes * in_channels, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        seq = self.proj(flatten_nodes(x))
        return self._head_from_sequence(self.encoder(seq))


class PatchTSTBaseline(BaselineBase):
    """Patch-based channel-independent Transformer-style baseline."""

    def __init__(self, num_nodes: int, in_channels: int = 1, d_model: int = 128, patch_len: int = 16, stride: int = 8, short_steps: int = 640, bp_steps: int = 1920) -> None:
        super().__init__(d_model, short_steps, bp_steps)
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(num_nodes * in_channels, d_model)
        self.encoder = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        seq = flatten_nodes(x)
        if seq.shape[1] < self.patch_len:
            patches = seq.mean(dim=1, keepdim=True)
        else:
            patches = seq.unfold(dimension=1, size=self.patch_len, step=self.stride).mean(dim=-1)
        h = self.encoder(self.proj(patches))
        return self._head_from_sequence(h)


class InvertedTransformerBaseline(BaselineBase):
    """iTransformer-style variate-token attention baseline."""

    def __init__(self, num_nodes: int, in_channels: int = 1, d_model: int = 128, num_layers: int = 3, num_heads: int = 4, short_steps: int = 640, bp_steps: int = 1920) -> None:
        super().__init__(d_model, short_steps, bp_steps)
        self.value_proj = nn.Linear(1, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        seq = flatten_nodes(x).transpose(1, 2).unsqueeze(-1)
        tokens = self.value_proj(seq.mean(dim=2, keepdim=True))[:, :, 0]
        h = self.encoder(tokens)
        return self._head_from_sequence(h)


class MixerBaseline(BaselineBase):
    """Multi-scale MLP mixer baseline used for TimeMixer/TimesNet-style wrappers."""

    def __init__(self, num_nodes: int, in_channels: int = 1, d_model: int = 128, short_steps: int = 640, bp_steps: int = 1920) -> None:
        super().__init__(d_model, short_steps, bp_steps)
        self.proj = nn.Linear(num_nodes * in_channels, d_model)
        self.mix = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        seq = self.proj(flatten_nodes(x))
        h = self.mix(seq) + seq
        return self._head_from_sequence(h)


class TimesNetBaseline(BaselineBase):
    """TimesNet-style baseline: FFT top-k period detection with 2D temporal convolution."""

    def __init__(self, num_nodes: int, in_channels: int = 1, d_model: int = 128, top_k: int = 3, short_steps: int = 640, bp_steps: int = 1920) -> None:
        super().__init__(d_model, short_steps, bp_steps)
        self.top_k = int(top_k)
        self.proj = nn.Linear(num_nodes * in_channels, d_model)
        self.conv2d = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
        )
        self.norm = nn.LayerNorm(d_model)

    def _periods(self, seq: torch.Tensor) -> list[int]:
        time = seq.shape[1]
        # cuFFT half precision requires power-of-two sizes; compute the spectrum in fp32.
        amp = torch.fft.rfft(seq.float(), dim=1).abs().mean(dim=(0, 2))
        if amp.numel() > 1:
            amp = amp.clone()
            amp[0] = 0.0
        k = min(self.top_k, amp.numel() - 1)
        freqs = torch.topk(amp, max(k, 1)).indices.tolist()
        periods = sorted({max(2, min(time, time // f)) for f in freqs if f > 0})
        return periods or [time]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        seq = self.proj(flatten_nodes(x))
        batch, time, dim = seq.shape
        res = seq
        periods = self._periods(seq)
        for period in periods:
            pad = (period - time % period) % period
            z = F.pad(seq, (0, 0, 0, pad)) if pad else seq
            rows = z.shape[1] // period
            z = z.reshape(batch, rows, period, dim).permute(0, 3, 1, 2)
            z = self.conv2d(z).permute(0, 2, 3, 1).reshape(batch, rows * period, dim)[:, :time]
            res = res + z
        return self._head_from_sequence(self.norm(res / (len(periods) + 1)))


class FourierBaseline(BaselineBase):
    """Frequency-domain baseline with low-mode reconstruction features."""

    def __init__(self, num_nodes: int, in_channels: int = 1, d_model: int = 128, modes: int = 32, short_steps: int = 640, bp_steps: int = 1920) -> None:
        super().__init__(d_model, short_steps, bp_steps)
        self.modes = modes
        self.proj = nn.Linear(num_nodes * in_channels, d_model)
        self.mix = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        seq = flatten_nodes(x)
        freq = torch.fft.rfft(seq, dim=1)
        keep = min(self.modes, freq.shape[1])
        filtered = torch.zeros_like(freq)
        filtered[:, :keep] = freq[:, :keep]
        recon = torch.fft.irfft(filtered, n=seq.shape[1], dim=1)
        return self._head_from_sequence(self.mix(self.proj(recon)))


class STGCNBaseline(BaselineBase):
    """Spatio-temporal GCN: gated temporal convolutions around a graph convolution."""

    def __init__(self, num_nodes: int, in_channels: int = 1, d_model: int = 128, kernel_size: int = 3, short_steps: int = 640, bp_steps: int = 1920) -> None:
        super().__init__(d_model, short_steps, bp_steps)
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.input_proj = nn.Linear(in_channels, d_model)
        self.adj_emb = nn.Parameter(torch.randn(num_nodes, d_model) * 0.02)
        pad = (kernel_size - 1) // 2
        self.t1 = nn.Conv1d(d_model, 2 * d_model, kernel_size, padding=pad)
        self.t2 = nn.Conv1d(d_model, 2 * d_model, kernel_size, padding=pad)
        self.spatial = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def _gated_temporal(self, conv: nn.Conv1d, h: torch.Tensor) -> torch.Tensor:
        batch, nodes, time, _ = h.shape
        z = h.reshape(batch * nodes, time, self.d_model).transpose(1, 2)
        z = F.glu(conv(z)[..., :time], dim=1)
        return z.transpose(1, 2).reshape(batch, nodes, time, self.d_model)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.input_proj(x)
        h = self._gated_temporal(self.t1, h)
        adj = torch.softmax(self.adj_emb @ self.adj_emb.T, dim=-1)
        h = F.gelu(self.spatial(torch.einsum("ij,bjtd->bitd", adj, h)))
        h = self._gated_temporal(self.t2, h)
        return self._head_from_sequence(self.norm(h.mean(dim=1)))


class GraphWaveNetBaseline(BaselineBase):
    """Graph WaveNet: dilated gated temporal convolutions with an adaptive adjacency."""

    def __init__(self, num_nodes: int, in_channels: int = 1, d_model: int = 128, dilations: tuple[int, ...] = (1, 2, 4, 8), adaptive_adj: bool = True, short_steps: int = 640, bp_steps: int = 1920) -> None:
        super().__init__(d_model, short_steps, bp_steps)
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.adaptive_adj = bool(adaptive_adj)
        self.input_proj = nn.Linear(in_channels, d_model)
        if self.adaptive_adj:
            self.e1 = nn.Parameter(torch.randn(num_nodes, 10) * 0.05)
            self.e2 = nn.Parameter(torch.randn(10, num_nodes) * 0.05)
        self.filters = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.gconv = nn.ModuleList()
        self.skips = nn.ModuleList()
        for d in dilations:
            self.filters.append(nn.Conv1d(d_model, d_model, 2, dilation=d, padding=d))
            self.gates.append(nn.Conv1d(d_model, d_model, 2, dilation=d, padding=d))
            self.gconv.append(nn.Linear(d_model, d_model))
            self.skips.append(nn.Linear(d_model, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, nodes, time, _ = x.shape
        h = self.input_proj(x)
        if self.adaptive_adj:
            adj = torch.softmax(F.relu(self.e1 @ self.e2), dim=-1)
        else:
            adj = torch.eye(nodes, device=x.device)
        skip_sum = h.new_zeros(batch, nodes, time, self.d_model)
        for filt, gate, gconv, skip in zip(self.filters, self.gates, self.gconv, self.skips):
            hin = h.reshape(batch * nodes, time, self.d_model).transpose(1, 2)
            tc = torch.tanh(filt(hin)[..., :time]) * torch.sigmoid(gate(hin)[..., :time])
            tc = tc.transpose(1, 2).reshape(batch, nodes, time, self.d_model)
            h = h + gconv(torch.einsum("ij,bjtd->bitd", adj, tc))
            skip_sum = skip_sum + skip(tc)
        return self._head_from_sequence(self.norm(skip_sum.mean(dim=1)))


class MTGNNBaseline(BaselineBase):
    """MTGNN: learned directed graph with top-k sparsification and mix-hop propagation."""

    def __init__(self, num_nodes: int, in_channels: int = 1, d_model: int = 128, mixhop_depth: int = 2, top_k: int = 4, short_steps: int = 640, bp_steps: int = 1920) -> None:
        super().__init__(d_model, short_steps, bp_steps)
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.mixhop_depth = int(mixhop_depth)
        self.top_k = min(int(top_k), num_nodes)
        self.alpha = 3.0
        self.input_proj = nn.Linear(in_channels, d_model)
        self.e1 = nn.Parameter(torch.randn(num_nodes, d_model) * 0.02)
        self.e2 = nn.Parameter(torch.randn(num_nodes, d_model) * 0.02)
        self.temporal = nn.Conv1d(d_model, d_model, kernel_size=6, padding=5)
        self.mix = nn.Linear(d_model * (self.mixhop_depth + 1), d_model)
        self.norm = nn.LayerNorm(d_model)

    def _graph(self) -> torch.Tensor:
        m1 = torch.tanh(self.alpha * self.e1)
        m2 = torch.tanh(self.alpha * self.e2)
        adj = F.relu(torch.tanh(self.alpha * (m1 @ m2.T - m2 @ m1.T)))
        if self.top_k < self.num_nodes:
            mask = torch.zeros_like(adj)
            idx = adj.topk(self.top_k, dim=-1).indices
            mask.scatter_(1, idx, 1.0)
            adj = adj * mask
        return adj / (adj.sum(-1, keepdim=True) + 1e-6)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, nodes, time, _ = x.shape
        h = self.input_proj(x)
        hin = h.reshape(batch * nodes, time, self.d_model).transpose(1, 2)
        h = F.gelu(self.temporal(hin)[..., :time]).transpose(1, 2).reshape(batch, nodes, time, self.d_model)
        adj = self._graph()
        outs = [h]
        cur = h
        for _ in range(self.mixhop_depth):
            cur = torch.einsum("ij,bjtd->bitd", adj, cur)
            outs.append(cur)
        h = self.mix(torch.cat(outs, dim=-1))
        return self._head_from_sequence(self.norm(h.mean(dim=1)))


class AGCRNBaseline(BaselineBase):
    """AGCRN: adaptive graph convolutional recurrent network over a pooled sequence."""

    def __init__(self, num_nodes: int, in_channels: int = 1, d_model: int = 128, embed_dim: int = 32, seq_pool: int = 64, short_steps: int = 640, bp_steps: int = 1920) -> None:
        super().__init__(d_model, short_steps, bp_steps)
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.seq_pool = int(seq_pool)
        self.node_emb = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.05)
        self.input_proj = nn.Linear(in_channels, d_model)
        self.cell = nn.GRUCell(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, nodes, time, _ = x.shape
        h = self.input_proj(x).permute(0, 1, 3, 2).reshape(batch * nodes, self.d_model, time)
        length = min(self.seq_pool, time)
        h = F.adaptive_avg_pool1d(h, length).reshape(batch, nodes, self.d_model, length).permute(0, 1, 3, 2)
        adj = torch.softmax(F.relu(self.node_emb @ self.node_emb.T), dim=-1)
        state = h.new_zeros(batch * nodes, self.d_model)
        outs = []
        for t in range(length):
            xt = torch.einsum("ij,bjd->bid", adj, h[:, :, t]).reshape(batch * nodes, self.d_model)
            state = self.cell(xt, state)
            outs.append(state.reshape(batch, nodes, self.d_model))
        seq = torch.stack(outs, dim=2).mean(dim=1)
        return self._head_from_sequence(self.norm(seq))


class CrossGNNBaseline(BaselineBase):
    """CrossGNN: stacked cross-variable attention and cross-scale temporal mixing."""

    def __init__(self, num_nodes: int, in_channels: int = 1, d_model: int = 128, num_layers: int = 3, short_steps: int = 640, bp_steps: int = 1920) -> None:
        super().__init__(d_model, short_steps, bp_steps)
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.scale = d_model ** -0.5
        self.input_proj = nn.Linear(in_channels, d_model)
        self.layers = nn.ModuleList(
            nn.ModuleDict(
                {
                    "q": nn.Linear(d_model, d_model),
                    "k": nn.Linear(d_model, d_model),
                    "v": nn.Linear(d_model, d_model),
                    "time": nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                    "ff": nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU()),
                }
            )
            for _ in range(int(num_layers))
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch, nodes, time, _ = x.shape
        h = self.input_proj(x)
        for layer in self.layers:
            q = layer["q"](h).mean(dim=2)
            k = layer["k"](h).mean(dim=2)
            v = layer["v"](h)
            attn = torch.softmax(q @ k.transpose(1, 2) * self.scale, dim=-1)
            h = h + torch.einsum("bij,bjtd->bitd", attn, v)
            hin = h.reshape(batch * nodes, time, self.d_model).transpose(1, 2)
            h = h + layer["time"](hin)[..., :time].transpose(1, 2).reshape(batch, nodes, time, self.d_model)
            h = h + layer["ff"](h)
        return self._head_from_sequence(self.norm(h.mean(dim=1)))


class CNNLSTMBaseline(BaselineBase):
    """CNN plus LSTM wearable-specific baseline."""

    def __init__(self, num_nodes: int, in_channels: int = 1, hidden_size: int = 128, short_steps: int = 640, bp_steps: int = 1920) -> None:
        super().__init__(hidden_size, short_steps, bp_steps)
        self.conv = nn.Sequential(
            nn.Conv1d(num_nodes * in_channels, hidden_size, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        seq = flatten_nodes(x).transpose(1, 2)
        seq = self.conv(seq).transpose(1, 2)
        h, _ = self.lstm(seq)
        return self._head_from_sequence(h)


class DeepPPGBaseline(BaselineBase):
    """Deep dilated residual CNN for PPG, distinct from CorNET's CNN-LSTM (no recurrence)."""

    def __init__(self, num_nodes: int, in_channels: int = 1, hidden_size: int = 128, num_blocks: int = 4, short_steps: int = 640, bp_steps: int = 1920) -> None:
        super().__init__(hidden_size, short_steps, bp_steps)
        self.stem = nn.Conv1d(num_nodes * in_channels, hidden_size, kernel_size=7, padding=3)
        self.blocks = nn.ModuleList()
        for i in range(int(num_blocks)):
            dilation = 2 ** i
            self.blocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=dilation, padding=dilation),
                    nn.GELU(),
                    nn.Conv1d(hidden_size, hidden_size, kernel_size=3, dilation=dilation, padding=dilation),
                    nn.GELU(),
                )
            )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.stem(flatten_nodes(x).transpose(1, 2))
        for block in self.blocks:
            h = h + block(h)
        return self._head_from_sequence(self.norm(h.transpose(1, 2)))


class BPNetBaseline(BaselineBase):
    """Simple CNN regressor for ECG/PPG to BP tasks."""

    def __init__(self, num_nodes: int, in_channels: int = 1, hidden_size: int = 128, short_steps: int = 640, bp_steps: int = 1920) -> None:
        super().__init__(hidden_size, short_steps, bp_steps)
        self.net = nn.Sequential(
            nn.Conv1d(num_nodes * in_channels, hidden_size, kernel_size=9, padding=4),
            nn.GELU(),
            nn.Conv1d(hidden_size, hidden_size, kernel_size=9, padding=4),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.net(flatten_nodes(x).transpose(1, 2)).transpose(1, 2)
        return self._head_from_sequence(h)
