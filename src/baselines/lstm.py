"""LSTM baseline wrapper."""

from __future__ import annotations

from src.baselines.common import LSTMBaseline


class LSTM(LSTMBaseline):
    """Thin wrapper exposing the shared baseline interface."""

    pass
