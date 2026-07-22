"""Tests for fmri_cognition.evaluate -- metrics on known arrays only."""
import numpy as np
import pytest

from fmri_cognition.evaluate import compute_metrics


def test_compute_metrics_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    metrics = compute_metrics(y, y)
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.0)


def test_compute_metrics_known_mae():
    y_true = np.array([0.0, 0.0, 0.0, 0.0])
    y_pred = np.array([1.0, 1.0, 1.0, 1.0])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["mae"] == pytest.approx(1.0)


def test_compute_metrics_shape_mismatch_raises():
    with pytest.raises(ValueError):
        compute_metrics(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
