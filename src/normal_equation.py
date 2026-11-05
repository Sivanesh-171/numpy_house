"""Closed-form Ordinary Least Squares solution via the Normal Equation.

Solves the linear regression objective analytically without gradient descent:
    theta = (X^T X)^(-1) X^T y
using numerically stable linear system solvers (np.linalg.solve / np.linalg.lstsq).
"""

from __future__ import annotations
from typing import Optional
import numpy as np


class NormalEquationRegression:
    """Analytical Ordinary Least Squares (OLS) solver using the Normal Equation.

    Educational Derivation:
    -----------------------
    The cost function is:
        J(theta) = (1 / 2m) * (X theta - y)^T (X theta - y)

    Taking the gradient with respect to theta and setting it to zero:
        nabla_theta J(theta) = (1 / m) * X^T (X theta - y) = 0
        => X^T X theta - X^T y = 0
        => (X^T X) theta = X^T y
        => theta = (X^T X)^(-1) X^T y

    Numerical Stability Note:
    -------------------------
    Directly computing the matrix inverse (np.linalg.inv) of X^T X is computationally
    wasteful O(n^3) and numerically fragile if X has collinear features.
    Instead, we solve the symmetric positive semi-definite linear system:
        A theta = b   where A = X^T X and b = X^T y
    using `np.linalg.solve`, falling back to `np.linalg.lstsq` (SVD-based least squares)
    if the matrix is singular.

    Attributes
    ----------
    theta_ : np.ndarray of shape (n_features + 1,)
        Learned parameters (including intercept).
    """

    def __init__(self) -> None:
        self.theta_: Optional[np.ndarray] = None

    def fit(self, X_bias: np.ndarray, y: np.ndarray) -> "NormalEquationRegression":
        """Fit linear regression via the normal equation.

        Parameters
        ----------
        X_bias : np.ndarray of shape (m, n_features + 1)
            Design matrix with intercept column already prepended.
        y : np.ndarray of shape (m,) or (m, 1)
            Target values.

        Returns
        -------
        self : NormalEquationRegression
            Fitted estimator.
        """
        X = np.asarray(X_bias, dtype=np.float64)
        y_vec = np.asarray(y, dtype=np.float64).ravel()

        if X.shape[0] != y_vec.shape[0]:
            raise ValueError(f"Sample count mismatch: X has {X.shape[0]} rows, y has {y_vec.shape[0]} elements.")

        # Form the normal equation: (X^T X) theta = X^T y
        XT_X = X.T @ X
        XT_y = X.T @ y_vec

        try:
            # np.linalg.solve uses LAPACK _gesv (LU decomposition)
            self.theta_ = np.linalg.solve(XT_X, XT_y)
        except np.linalg.LinAlgError:
            # Fall back to pseudoinverse / least squares if XT_X is singular
            self.theta_, residuals, rank, s = np.linalg.lstsq(X, y_vec, rcond=None)

        return self

    def predict(self, X_bias: np.ndarray) -> np.ndarray:
        """Predict target values using the closed-form parameters.

        Parameters
        ----------
        X_bias : np.ndarray of shape (m, n_features + 1)
            Design matrix with intercept column included.

        Returns
        -------
        y_pred : np.ndarray of shape (m,)
            Predicted values.
        """
        if self.theta_ is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' before 'predict'.")

        X = np.asarray(X_bias, dtype=np.float64)
        return X @ self.theta_

