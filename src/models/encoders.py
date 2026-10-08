"""Temporal encoders for PI-CDGN."""

from __future__ import annotations

import torch
from torch import nn


class Chomp1d(nn.Module):
    """Remove right padding introduced for causal convolutions."""

    def __init__(self, chomp_size: int) -> None:
        """Create a chomp layer."""

        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Trim the final ``chomp_size`` time steps."""

        if self.chomp_size == 0:
            return x
        return x[..., : -self.chomp_size]


class CausalConvBlock(nn.Module):
    """Residual dilated causal convolution block."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        """Initialize the block."""

        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual causal convolutions to ``(batch, channels, time)`` input."""

        y = self.net(x) + x
        return self.norm(y.transpose(1, 2)).transpose(1, 2)


class NodeTemporalEncoder(nn.Module):
    """Per-node TCN plus causal Transformer temporal encoder."""

    def __init__(
        self,
        in_channels: int,
        d_model: int = 128,
        kernel_size: int = 5,
        dilations: tuple[int, ...] = (1, 2, 4, 8),
        transformer_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        downsample_factor: int = 8,
    ) -> None:
        """Create the encoder.

        Args:
            in_channels: Per-node feature dimension in the input tensor.
            d_model: Hidden embedding dimension.
            kernel_size: Causal convolution kernel size.
            dilations: TCN dilation schedule.
            transformer_layers: Number of Transformer encoder layers.
            num_heads: Transformer attention heads.
            dropout: Dropout probability.
        """

        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=1)
        self.downsample_factor = max(1, int(downsample_factor))
        self.tcn = nn.Sequential(
            *[CausalConvBlock(d_model, kernel_size, dilation, dropout) for dilation in dilations]
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=transformer_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input windows.

        Args:
            x: Tensor with shape ``(batch, nodes, time, features)``.

        Returns:
            Tensor with shape ``(batch, nodes, time, d_model)``.
        """

        if x.ndim != 4:
            raise ValueError(f"Expected x shape (batch, nodes, time, features), got {tuple(x.shape)}")
        batch, nodes, time, features = x.shape
        z = x.reshape(batch * nodes, time, features).transpose(1, 2)
        z = self.input_proj(z)
        z = self.tcn(z)
        if self.downsample_factor > 1:
            z = torch.nn.functional.avg_pool1d(z, kernel_size=self.downsample_factor, stride=self.downsample_factor)
        z = z.transpose(1, 2)
        encoded_time = z.shape[1]
        causal_mask = torch.triu(
            torch.ones(encoded_time, encoded_time, device=x.device, dtype=torch.bool), diagonal=1
        )
        z = self.transformer(z, mask=causal_mask)
        return z.reshape(batch, nodes, encoded_time, -1)


class IdentityTemporalEncoder(nn.Module):
    """Minimal encoder preserving the node-time interface."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` unchanged."""

        return x
