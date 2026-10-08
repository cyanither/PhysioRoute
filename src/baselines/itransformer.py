"""ITransformer baseline wrapper."""

from __future__ import annotations

from src.baselines.common import InvertedTransformerBaseline


class ITransformer(InvertedTransformerBaseline):
    """Thin wrapper exposing the shared baseline interface."""

    pass
