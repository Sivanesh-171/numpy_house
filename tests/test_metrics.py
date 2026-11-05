"""Tests for regression evaluation metrics (MSE, RMSE, MAE, R^2)."""

import numpy as np
import pytest
from src.metrics import mse, rmse, mae, r2_score


def test_hand_computed_toy_example():
    """Verify MSE and R^2 against exact hand-calculated pencil-and-paper values.

    Ground truth y: [2.0, 4.0, 5.0, 7.0]
    Predictions y_hat: [2.5, 3.5, 5.0, 8.0]

    Hand calculation:
    -----------------
    Errors (y - y_hat) = [-0.5, 0.5, 0.0, -1.0]
    Squared errors     = [0.25, 0.25, 0.0, 1.00]
    Sum of squared err = 1.50
    m = 4
    MSE = 1.50 / 4 = 0.375

    Mean of y = (2 + 4 + 5 + 7) / 4 = 18 / 4 = 4.5
    Total dev (y - y_bar) = [-2.5, -0.5, 0.5, 2.5]
    Squared dev           = [6.25, 0.25, 0.25, 6.25]
    SS_tot = 13.00
    SS_res = 1.50
    R^2 = 1 - (1.50 / 13.00) = 1 - 3/26 = 23/26 ~= 0.8846153846153846
    """
    y_true = np.array([2.0, 4.0, 5.0, 7.0])
    y_pred = np.array([2.5, 3.5, 5.0, 8.0])

    expected_mse = 0.375
    expected_r2 = 23.0 / 26.0

    calc_mse = mse(y_true, y_pred)
    calc_r2 = r2_score(y_true, y_pred)

    assert np.isclose(calc_mse, expected_mse, atol=1e-12), f"Expected MSE {expected_mse}, got {calc_mse}"
    assert np.isclose(calc_r2, expected_r2, atol=1e-12), f"Expected R2 {expected_r2}, got {calc_r2}"


def test_perfect_prediction():
    """Verify that perfect predictions yield MSE = 0.0 and R^2 = 1.0."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([10.0, 20.0, 30.0, 40.0])

    assert np.isclose(mse(y_true, y_pred), 0.0, atol=1e-12)
    assert np.isclose(rmse(y_true, y_pred), 0.0, atol=1e-12)
    assert np.isclose(mae(y_true, y_pred), 0.0, atol=1e-12)
    assert np.isclose(r2_score(y_true, y_pred), 1.0, atol=1e-12)


def test_mean_baseline_prediction():
    """Verify that predicting the mean of y yields R^2 = 0.0."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_mean = np.mean(y_true)
    y_pred = np.full_like(y_true, fill_value=y_mean)

    assert np.isclose(r2_score(y_true, y_pred), 0.0, atol=1e-12)


def test_worse_than_mean_prediction():
    """Verify that wild predictions yield negative R^2."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([100.0, 200.0, 300.0])

    assert r2_score(y_true, y_pred) < 0.0


def test_shape_mismatch_raises():
    """Verify ValueError is raised on mismatched input dimensions."""
    with pytest.raises(ValueError):
        mse(np.array([1.0, 2.0]), np.array([1.0]))

    with pytest.raises(ValueError):
        r2_score(np.array([1.0, 2.0]), np.array([1.0]))

