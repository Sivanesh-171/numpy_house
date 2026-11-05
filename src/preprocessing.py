"""Data preprocessing utilities written from scratch using pure NumPy.

Includes:
- StandardScaler: Z-score feature standardization (strictly fitted on training data).
- add_bias: Appends an intercept/bias column of ones to the design matrix.
"""

from __future__ import annotations
from typing import Optional, Union, Tuple
import numpy as np


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:
        z = (x - u) / s

    where `u` is the mean of the training samples, and `s` is the standard
    deviation of the training samples.

    Educational Note:
    -----------------
    In gradient descent, feature scaling is crucial. If features have vastly
    different scales (e.g., HouseAge ~ 10-50 vs Population ~ 1000-35000), the
    cost function contours become highly elongated ellipses. The gradient
    oscillates inefficiently perpendicular to the minimum. Standardization transforms
    the cost contours into concentric circles/spheres, allowing gradient descent
    to take a direct, rapid path toward the global minimum with a higher learning rate.

    Attributes
    ----------
    mean_ : np.ndarray of shape (n_features,)
        The mean value for each feature in the training set.
    scale_ : np.ndarray of shape (n_features,)
        The standard deviation (scale) for each feature in the training set.
        Zero standard deviations are guarded and replaced with 1.0 to avoid division by zero.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(self) -> None:
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute the mean and standard deviation from training data X.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) or (n_samples,)
            The training data used to compute the mean and standard deviation.

        Returns
        -------
        self : StandardScaler
            Fitted scaler.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        self.n_features_in_ = X_arr.shape[1]

        # Compute sample statistics along sample axis (axis 0)
        # Using ddof=0 matches scikit-learn's population std formula: sqrt(sum((x - mean)^2) / m)
        self.mean_ = np.mean(X_arr, axis=0)
        std = np.std(X_arr, axis=0, ddof=0)

        # Numerical stability guard:
        # If a feature is constant (zero variance), std is 0.
        # Dividing by zero would produce NaN/inf. We replace zero or near-zero std with 1.0,
        # which centers the feature at 0.0 without exploding.
        eps = 1e-12
        self.scale_ = np.where(std < eps, 1.0, std)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Perform standardization by centering and scaling.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features) or (n_samples,)
            The data to transform.

        Returns
        -------
        X_scaled : np.ndarray
            Standardized array.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler instance is not fitted yet. Call 'fit' before 'transform'.")

        X_arr = np.asarray(X, dtype=np.float64)
        was_1d = (X_arr.ndim == 1)
        if was_1d:
            X_arr = X_arr.reshape(-1, 1)

        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: Scaler expected {self.n_features_in_} features, "
                f"but received {X_arr.shape[1]}."
            )

        # Vectorized broadcasting: (n_samples, n_features) - (n_features,) / (n_features,)
        X_scaled = (X_arr - self.mean_) / self.scale_

        return X_scaled.ravel() if was_1d else X_scaled

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it.

        Parameters
        ----------
        X : np.ndarray
            Data to fit and transform.

        Returns
        -------
        X_scaled : np.ndarray
            Transformed data.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        """Scale back the data to the original representation.

        Parameters
        ----------
        X_scaled : np.ndarray
            The standardized data.

        Returns
        -------
        X_orig : np.ndarray
            Data restored to original scale: X_orig = X_scaled * scale + mean.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("StandardScaler instance is not fitted yet.")

        X_arr = np.asarray(X_scaled, dtype=np.float64)
        was_1d = (X_arr.ndim == 1)
        if was_1d:
            X_arr = X_arr.reshape(-1, 1)

        X_orig = (X_arr * self.scale_) + self.mean_
        return X_orig.ravel() if was_1d else X_orig


def add_bias(X: np.ndarray, position: str = "first") -> np.ndarray:
    """Prepend or append a column of ones (bias/intercept term x0 = 1) to the feature matrix.

    Educational Note:
    -----------------
    By augmenting the feature matrix X with an intercept column of ones:
        x_0 = 1,
    the linear hypothesis:
        h_theta(x) = theta_0 + theta_1 * x_1 + ... + theta_n * x_n
    can be written compactly in vectorized matrix form:
        h_theta(X) = X @ theta
    where theta = [theta_0, theta_1, ..., theta_n]^T.
    This eliminates special-case handling for the intercept theta_0 in both the
    cost and gradient calculations.

    IMPORTANT: In standard regression pipelines, bias is added AFTER feature scaling,
    because a column of constant ones has zero standard deviation and cannot be standardized.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features) or (n_samples,)
        Feature matrix.
    position : str, default="first"
        Position where the bias column is added:
        - "first": column 0 is ones (theta[0] is the intercept).
        - "last": column -1 is ones (theta[-1] is the intercept).

    Returns
    -------
    X_bias : np.ndarray of shape (n_samples, n_features + 1)
        Design matrix augmented with a column of ones.
    """
    X_arr = np.asarray(X, dtype=np.float64)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)

    m = X_arr.shape[0]
    ones = np.ones((m, 1), dtype=np.float64)

    if position == "first":
        return np.hstack((ones, X_arr))
    elif position == "last":
        return np.hstack((X_arr, ones))
    else:
        raise ValueError(f"Invalid position '{position}'. Must be 'first' or 'last'.")


def unstandardize_coefficients(
    theta: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray
) -> tuple[float, np.ndarray]:
    """Convert parameters learned on standardized features back to original feature scale.

    Educational Derivation:
    -----------------------
    In standardized space:
        y_hat = theta_0 + sum_{j=1}^n theta_j * ((x_j - mu_j) / sigma_j)
    Expanding terms:
        y_hat = (theta_0 - sum_{j=1}^n (theta_j * mu_j / sigma_j)) + sum_{j=1}^n (theta_j / sigma_j) * x_j

    Therefore, the original unstandardized coefficients are:
        beta_j = theta_j / sigma_j
    and the unstandardized intercept is:
        beta_0 = theta_0 - sum_{j=1}^n beta_j * mu_j

    Parameters
    ----------
    theta : np.ndarray of shape (n_features + 1,)
        Learned parameters where theta[0] is the intercept and theta[1:] are feature weights.
    mean : np.ndarray of shape (n_features,)
        Training means used by StandardScaler.
    scale : np.ndarray of shape (n_features,)
        Training standard deviations used by StandardScaler.

    Returns
    -------
    beta_0 : float
        Unstandardized intercept in original units.
    beta : np.ndarray of shape (n_features,)
        Unstandardized coefficients in original feature units.
    """
    theta_0 = float(theta[0])
    theta_weights = np.asarray(theta[1:], dtype=np.float64)

    mu = np.asarray(mean, dtype=np.float64)
    sigma = np.asarray(scale, dtype=np.float64)

    # beta_j = theta_j / sigma_j
    beta = theta_weights / sigma

    # beta_0 = theta_0 - sum(beta_j * mu_j)
    beta_0 = theta_0 - float(np.sum(beta * mu))

    return beta_0, beta

