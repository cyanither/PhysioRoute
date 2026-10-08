from __future__ import annotations

import torch

from src.baselines import BASELINE_REGISTRY


def test_all_baseline_forward_shapes() -> None:
    x = torch.randn(2, 3, 24, 1)
    for name, cls in BASELINE_REGISTRY.items():
        model = cls(num_nodes=3, in_channels=1, short_steps=4, bp_steps=5)
        out = model(x)
        assert out["wave"].shape == (2, 4), name
        assert out["hr"].shape == (2, 1), name
        assert out["bp"].shape == (2, 5, 2), name
        assert out["stress"].shape == (2, 1), name
