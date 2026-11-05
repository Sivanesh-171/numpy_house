"""Evaluation metrics for regression written from scratch using pure NumPy.

Includes:
- mse: Mean Squared Error (reported test MSE without 1/2 factor)
- rmse: Root Mean Squared Error
- mae: Mean Absolute Error
- r2_score: Coefficient of Determination (R^2 = 1 - SS_res / SS_tot)
"""

from __future__ import annotations
import numpy as np


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute the Mean Squared Error (MSE).

    Mathematical Formulation:
    -------------------------
        MSE = (1 / m) * sum_{i=1}^{m} (y_i - y_hat_i)^2

    Educational Distinction:
    -----------------------
    In the gradient descent cost function J(theta), a factor of 1/2 is included
    to cancel out the power 2 when taking the derivative (d/d_theta).
    However, standard reported MSE (as in scikit-learn and standard literature)
    does NOT include the 1/2 factor.

    Parameters
    ----------
    y_true : np.ndarray of shape (m,) or (m, 1)
        Ground truth (correct) target values.
    y_pred : np.ndarray of shape (m,) or (m, 1)
        Estimated target values.

    Returns
    -------
    float
        The calculated Mean Squared Error.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError(f"Shape mismatch: y_true has {y_t.shape[0]} samples, y_pred has {y_p.shape[0]} samples.")
    if y_t.shape[0] == 0:
        raise ValueError("Cannot compute MSE on empty arrays.")

    residuals = y_t - y_p
    # Vectorized dot product of residuals is numerically faster and cleaner than sum(residuals ** 2)
    return float(np.mean(residuals ** 2))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE) in original target units."""
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE)."""
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean(np.abs(y_t - y_p)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Coefficient of Determination (R^2 score).

    Mathematical Formulation:
    -------------------------
        SS_res = sum_{i=1}^m (y_i - y_hat_i)^2  (Residual Sum of Squares)
        SS_tot = sum_{i=1}^m (y_i - y_bar)^2    (Total Sum of Squares)
        R^2 = 1 - (SS_res / SS_tot)

    Interpretation:
    ---------------
    - R^2 = 1.0 indicates perfect predictions.
    - R^2 = 0.0 means the model performs no better than simply predicting the mean of y.
    - R^2 < 0.0 indicates a model that performs worse than predicting the mean of y.

    Parameters
    ----------
    y_true : np.ndarray of shape (m,) or (m, 1)
        Ground truth (correct) target values.
    y_pred : np.ndarray of shape (m,) or (m, 1)
        Estimated target values.

    Returns
    -------
    float
        The calculated R^2 score.
    """
    y_t = np.asarray(y_true, dtype=np.float64).ravel()
    y_p = np.asarray(y_pred, dtype=np.float64).ravel()

    if y_t.shape[0] != y_p.shape[0]:
        raise ValueError(f"Shape mismatch: y_true has {y_t.shape[0]} samples, y_pred has {y_p.shape[0]} samples.")
    if y_t.shape[0] == 0:
        raise ValueError("Cannot compute R2 score on empty arrays.")

    ss_res = np.sum((y_t - y_p) ** 2)
    y_bar = np.mean(y_t)
    ss_tot = np.sum((y_t - y_bar) ** 2)

    # Guard against zero total variance (constant target)
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    return float(1.0 - (ss_res / ss_tot))

