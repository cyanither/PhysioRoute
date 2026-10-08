from __future__ import annotations

import torch

from src.models.graph_learner import gaussian_soft_delay


def test_gaussian_soft_delay_shape_and_grad() -> None:
    h = torch.randn(2, 8, 3, requires_grad=True)
    delays = torch.full((2,), 0.125, requires_grad=True)
    out = gaussian_soft_delay(h, delays, fs=8.0, sigma=0.75)
    assert out.shape == h.shape
    loss = out.square().mean()
    loss.backward()
    assert h.grad is not None
    assert delays.grad is not None
    assert torch.isfinite(delays.grad).all()
