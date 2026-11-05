"""Tests for custom StandardScaler, add_bias, and coefficient unstandardization."""

import numpy as np
import pytest
from src.preprocessing import StandardScaler, add_bias, unstandardize_coefficients


def test_scaler_no_test_leakage():
    """Verify that scaler parameters come strictly from training data and test data does not leak."""
    # Training data: mean = 10.0, std = 2.0
    X_train = np.array([[8.0, 100.0],
                        [10.0, 200.0],
                        [12.0, 300.0]])

    # Extreme test data with radically different distribution (mean = 10000, std = 5000)
    X_test = np.array([[10000.0, 999999.0],
                       [50000.0, 888888.0]])

    scaler = StandardScaler()
    scaler.fit(X_train)

    train_mean_expected = np.mean(X_train, axis=0)
    train_std_expected = np.std(X_train, axis=0, ddof=0)

    # Assert scaler parameters match training set
    np.testing.assert_allclose(scaler.mean_, train_mean_expected, atol=1e-12)
    np.testing.assert_allclose(scaler.scale_, train_std_expected, atol=1e-12)

    # Transform test set
    X_test_scaled = scaler.transform(X_test)

    # Crucial assertion: transforming test data must NOT change scaler's internal mean or scale
    np.testing.assert_allclose(scaler.mean_, train_mean_expected, atol=1e-12)
    np.testing.assert_allclose(scaler.scale_, train_std_expected, atol=1e-12)

    # Test set transformation must use strictly train mean and std: (X_test - train_mean) / train_std
    expected_test_scaled = (X_test - train_mean_expected) / train_std_expected
    np.testing.assert_allclose(X_test_scaled, expected_test_scaled, atol=1e-12)


def test_scaler_zero_std_guard():
    """Verify that zero-variance (constant) features do not cause division by zero or NaN."""
    # Column 0 has variance, Column 1 is strictly constant (zero std)
    X = np.array([[1.0, 5.0],
                  [2.0, 5.0],
                  [3.0, 5.0]])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    assert not np.isnan(X_scaled).any(), "StandardScaler produced NaNs on zero-variance feature!"
    assert not np.isinf(X_scaled).any(), "StandardScaler produced Infs on zero-variance feature!"

    # Scaled constant column should be centered at 0.0: (5.0 - 5.0) / 1.0 = 0.0
    np.testing.assert_allclose(X_scaled[:, 1], [0.0, 0.0, 0.0], atol=1e-12)


def test_add_bias_first_and_last():
    """Verify add_bias correctly augments the design matrix with ones."""
    X = np.array([[2.0, 3.0],
                  [4.0, 5.0]])

    X_first = add_bias(X, position="first")
    expected_first = np.array([[1.0, 2.0, 3.0],
                               [1.0, 4.0, 5.0]])
    np.testing.assert_allclose(X_first, expected_first, atol=1e-12)

    X_last = add_bias(X, position="last")
    expected_last = np.array([[2.0, 3.0, 1.0],
                              [4.0, 5.0, 1.0]])
    np.testing.assert_allclose(X_last, expected_last, atol=1e-12)


def test_unstandardize_coefficients_reconstruction():
    """Verify that unstandardized coefficients yield identical predictions on original data."""
    np.random.seed(42)
    m, n = 50, 3
    X_raw = np.random.randn(m, n) * 10 + 25  # Raw unscaled data
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # Synthetic weights in standardized space: [intercept, w1, w2, w3]
    theta = np.array([3.5, 1.2, -0.8, 2.1])

    # Prediction in standardized space:
    X_bias = add_bias(X_scaled, position="first")
    y_pred_scaled = X_bias @ theta

    # Unstandardize coefficients to original units
    beta_0, beta = unstandardize_coefficients(theta, scaler.mean_, scaler.scale_)

    # Prediction on raw original data:
    y_pred_raw = beta_0 + (X_raw @ beta)

    # Both predictions must match exactly up to floating-point tolerance
    np.testing.assert_allclose(y_pred_raw, y_pred_scaled, atol=1e-10)

