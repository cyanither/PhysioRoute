"""CorNET baseline wrapper."""

from __future__ import annotations

from src.baselines.common import CNNLSTMBaseline


class CorNET(CNNLSTMBaseline):
    """Thin wrapper exposing the shared baseline interface."""

    pass
