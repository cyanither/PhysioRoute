from __future__ import annotations

import numpy as np

from src.eval import bp_metrics, regression_metrics, waveform_metrics


def test_waveform_metrics() -> None:
    metrics = waveform_metrics(np.array([[1.0, 2.0]]), np.array([[1.0, 3.0]]))
    assert metrics["mse"] == 0.5


def test_regression_metrics() -> None:
    metrics = regression_metrics(np.array([[80.0], [90.0]]), np.array([[70.0], [100.0]]), "hr")
    assert metrics["hr_mae"] == 10.0


def test_bp_metrics() -> None:
    pred = np.array([[[120.0, 80.0]]])
    target = np.array([[[124.0, 92.0]]])
    metrics = bp_metrics(pred, target)
    assert metrics["sbp_acc_5"] == 1.0
    assert metrics["dbp_acc_10"] == 0.0
