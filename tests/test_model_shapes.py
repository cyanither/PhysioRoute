from __future__ import annotations

import torch

from src.models.graph_learner import PhysiologyGraphLearner
from src.models.picdgn import PICDGN


def test_graph_learner_output_shapes() -> None:
    learner = PhysiologyGraphLearner(num_nodes=3, d_model=8, chunk_size=4)
    h = torch.randn(2, 3, 10, 8)
    out = learner(h)
    assert out.edge_weight.shape == (2, 3, 3, 3)
    assert out.delay.shape == (2, 3, 3, 3)
    diagonal = torch.diagonal(out.edge_weight, dim1=-2, dim2=-1)
    assert torch.count_nonzero(diagonal) == 0


def test_picdgn_forward_shapes() -> None:
    model = PICDGN(num_nodes=3, in_channels=1, d_model=8, target_fs=8, chunk_sec=1, short_steps=4, bp_steps=5, message_passing_steps=1)
    out = model(torch.randn(2, 3, 16, 1))
    assert out["wave"].shape == (2, 4)
    assert out["hr"].shape == (2, 1)
    assert out["bp"].shape == (2, 5, 2)
    assert out["stress"].shape == (2, 1)


def test_pairwise_graph_does_not_use_unrelated_node_state() -> None:
    learner = PhysiologyGraphLearner(num_nodes=3, d_model=8, chunk_size=4)
    learner.eval()
    base = torch.zeros(1, 3, 8, 8)
    changed = base.clone()
    changed[:, 0] = 1.0
    out_base = learner(base).edge_weight
    out_changed = learner(changed).edge_weight
    assert torch.allclose(out_base[..., 1, 2], out_changed[..., 1, 2])
    assert not torch.allclose(out_base[..., 0, 1], out_changed[..., 0, 1])


def test_delay_head_can_start_from_training_pat_prior() -> None:
    learner = PhysiologyGraphLearner(
        num_nodes=3,
        d_model=8,
        chunk_size=4,
        tau_min=0.0,
        tau_max=0.5,
        delay_init=0.16,
    )
    learner.eval()
    out = learner(torch.randn(2, 3, 8, 8))
    off_diagonal = ~torch.eye(3, dtype=torch.bool)
    assert torch.allclose(
        out.delay[..., off_diagonal],
        torch.full_like(out.delay[..., off_diagonal], 0.16),
        atol=1e-6,
    )
