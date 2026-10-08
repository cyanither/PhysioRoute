"""Transformer baseline wrapper."""

from __future__ import annotations

from src.baselines.common import TransformerBaseline


class Transformer(TransformerBaseline):
    """Thin wrapper exposing the shared baseline interface."""

    pass
