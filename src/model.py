"""Linear Regression with Batch Gradient Descent implemented from scratch using NumPy.

Contains:
- LinearRegressionGD: Vectorized batch gradient descent optimizer for linear regression.
"""

from __future__ import annotations
from typing import Optional, List
import warnings
import numpy as np


class DivergenceWarning(RuntimeWarning):
    """Warning issued when gradient descent exhibits numerical instability or cost increases."""
    pass


class LinearRegressionGD:
    """Linear Regression trained via Batch Gradient Descent using pure NumPy.

    Mathematical Formulation:
    -------------------------
    1. Hypothesis:
       The linear model makes predictions as a linear combination of input features:
           h_theta(x) = theta_0 * 1 + theta_1 * x_1 + ... + theta_n * x_n
       In vectorized matrix notation (where X includes the intercept column x_0 = 1):
           h_theta(X) = X @ theta
       where:
           X in R^{m x (n+1)} is the augmented design matrix
           theta in R^{(n+1)} is the parameter weight vector
           h_theta(X) in R^m is the predicted target vector

    2. Cost Function (Mean Squared Error with 1/2 factor):
           J(theta) = (1 / (2 * m)) * sum_{i=1}^m (h_theta(x^{(i)}) - y^{(i)})^2
       In vectorized form:
           error = X @ theta - y
           J(theta) = (1 / (2 * m)) * (error^T @ error)

    3. Analytic Gradient:
       The gradient is the vector of partial derivatives of J(theta) with respect to each parameter theta_j:
           d J(theta) / d theta_j = (1 / m) * sum_{i=1}^m (h_theta(x^{(i)}) - y^{(i)}) * x^{(i)}_j
       In vectorized matrix form:
           nabla_theta J(theta) = (1 / m) * X^T @ (X @ theta - y)

    4. Parameter Update Rule:
       At each iteration, we step opposite to the gradient:
           theta^{(t+1)} = theta^{(t)} - alpha * nabla_theta J(theta^{(t)})
       where alpha is the learning rate.

    Parameters
    ----------
    alpha : float, default=0.01
        Learning rate (step size) for gradient descent.
    n_iters : int, default=1000
        Maximum number of gradient descent iterations.
    tol : float, optional
        Tolerance for early stopping. If |J^{(t)} - J^{(t-1)}| < tol,
        optimization stops early.
    raise_on_divergence : bool, default=False
        If True, raises a ValueError when divergence or NaN/inf occurs.
        If False, issues a DivergenceWarning and terminates gracefully.
    verbose : bool, default=False
        If True, prints cost progress periodically during training.

    Attributes
    ----------
    theta_ : np.ndarray of shape (n_features + 1,)
        Learned model weights (including intercept as theta_[0]).
    history_ : list of float
        History of cost function values J(theta) at each iteration.
    n_iters_ : int
        Actual number of iterations performed until convergence or termination.
    """

    def __init__(
        self,
        alpha: float = 0.01,
        n_iters: int = 1000,
        tol: Optional[float] = None,
        raise_on_divergence: bool = False,
        verbose: bool = False,
    ) -> None:
        self.alpha = float(alpha)
        self.n_iters = int(n_iters)
        self.tol = float(tol) if tol is not None else None
        self.raise_on_divergence = raise_on_divergence
        self.verbose = verbose

        self.theta_: Optional[np.ndarray] = None
        self.history_: List[float] = []
        self.n_iters_: int = 0

    @staticmethod
    def compute_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """Compute the Mean Squared Error cost J(theta) with the 1/2 factor.

        Mathematical formula:
            J(theta) = (1 / 2m) * ||X @ theta - y||^2

        Parameters
        ----------
        X : np.ndarray of shape (m, n_features + 1)
            Augmented design matrix (with intercept column).
        y : np.ndarray of shape (m,)
            Target vector.
        theta : np.ndarray of shape (n_features + 1,)
            Parameter vector.

        Returns
        -------
        cost : float
            Value of the cost function J(theta).
        """
        m = X.shape[0]
        error = (X @ theta) - y
        # Vectorized inner product error.T @ error
        cost = (1.0 / (2.0 * m)) * np.dot(error, error)
        return float(cost)

    @staticmethod
    def compute_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Compute the analytic gradient of J(theta) with respect to theta.

        Mathematical formula:
            nabla_theta J(theta) = (1 / m) * X^T @ (X @ theta - y)

        Parameters
        ----------
        X : np.ndarray of shape (m, n_features + 1)
            Augmented design matrix (with intercept column).
        y : np.ndarray of shape (m,)
            Target vector.
        theta : np.ndarray of shape (n_features + 1,)
            Parameter vector.

        Returns
        -------
        gradient : np.ndarray of shape (n_features + 1,)
            Analytic gradient vector.
        """
        m = X.shape[0]
        error = (X @ theta) - y
        # Matrix-vector multiplication: (n_features + 1, m) @ (m,) -> (n_features + 1,)
        gradient = (1.0 / m) * (X.T @ error)
        return gradient

    def fit(self, X_bias: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        """Fit linear regression model using batch gradient descent.

        Parameters
        ----------
        X_bias : np.ndarray of shape (m, n_features + 1)
            Standardized design matrix with the bias column of ones prepended.
        y : np.ndarray of shape (m,) or (m, 1)
            Target values.

        Returns
        -------
        self : LinearRegressionGD
            Fitted estimator.
        """
        X = np.asarray(X_bias, dtype=np.float64)
        y_vec = np.asarray(y, dtype=np.float64).ravel()

        m, n_params = X.shape
        if m != y_vec.shape[0]:
            raise ValueError(f"Sample count mismatch: X has {m} rows, y has {y_vec.shape[0]} elements.")

        # Initialize parameters to zeros
        self.theta_ = np.zeros(n_params, dtype=np.float64)
        self.history_ = []
        self.n_iters_ = 0
        warned_increase = False

        for iteration in range(self.n_iters):
            # 1. Compute current cost J(theta)
            current_cost = self.compute_cost(X, y_vec, self.theta_)

            # 2. Check for NaN or Inf (divergence check)
            if np.isnan(current_cost) or np.isinf(current_cost):
                msg = (
                    f"Gradient descent diverged at iteration {iteration}: "
                    f"cost became {current_cost}. Learning rate alpha={self.alpha} is too large."
                )
                if self.raise_on_divergence:
                    raise ValueError(msg)
                else:
                    warnings.warn(msg, DivergenceWarning)
                    self.history_.append(float("inf"))
                    self.n_iters_ = iteration + 1
                    break

            # 3. Check for cost increase (divergence warning)
            if iteration > 0:
                prev_cost = self.history_[-1]
                # Floating-point safety threshold: 1e-10
                if current_cost > prev_cost + 1e-10:
                    msg = (
                        f"Gradient descent cost increased from {prev_cost:.6f} to {current_cost:.6f} "
                        f"at iteration {iteration}. Alpha={self.alpha} may be too large."
                    )
                    if self.raise_on_divergence:
                        raise ValueError(msg)
                    elif not warned_increase:
                        warnings.warn(msg, DivergenceWarning)
                        warned_increase = True

            self.history_.append(current_cost)
            self.n_iters_ = iteration + 1

            if self.verbose and (iteration % max(1, self.n_iters // 10) == 0 or iteration == self.n_iters - 1):
                print(f"[GD Iter {iteration:4d}/{self.n_iters}] Cost J(theta) = {current_cost:.6f}")

            # 4. Check for early stopping convergence
            if self.tol is not None and iteration > 0:
                cost_improvement = abs(self.history_[-2] - current_cost)
                if cost_improvement < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {iteration} (cost improvement {cost_improvement:.2e} < tol {self.tol:.2e}).")
                    break

            # 5. Compute analytic gradient and perform gradient descent update
            gradient = self.compute_gradient(X, y_vec, self.theta_)
            self.theta_ = self.theta_ - (self.alpha * gradient)

        return self

    def predict(self, X_bias: np.ndarray) -> np.ndarray:
        """Predict target values using the learned parameters.

        Mathematical formula:
            y_pred = X @ theta

        Parameters
        ----------
        X_bias : np.ndarray of shape (m, n_features + 1)
            Augmented design matrix (with intercept column).

        Returns
        -------
        y_pred : np.ndarray of shape (m,)
            Predicted values.
        """
        if self.theta_ is None:
            raise RuntimeError("Model is not fitted yet. Call 'fit' before 'predict'.")

        X = np.asarray(X_bias, dtype=np.float64)
        return X @ self.theta_

    @property
    def intercept_(self) -> float:
        """Return the learned bias/intercept theta_0."""
        if self.theta_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return float(self.theta_[0])

    @property
    def coef_(self) -> np.ndarray:
        """Return the learned feature coefficients [theta_1, ..., theta_n]."""
        if self.theta_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.theta_[1:].copy()

