"""Tests verifying that analytic gradient matches numerical finite-difference approximation."""

import numpy as np
from src.model import LinearRegressionGD
from src.preprocessing import add_bias


def compute_numerical_gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    """Compute the two-sided central finite-difference gradient approximation of J(theta).

    Formula for each parameter j:
        grad_num_j = (J(theta + eps * e_j) - J(theta - eps * e_j)) / (2 * eps)
    where e_j is the standard unit basis vector along dimension j.
    """
    grad_num = np.zeros_like(theta, dtype=np.float64)
    n_params = len(theta)

    for j in range(n_params):
        theta_plus = theta.copy()
        theta_plus[j] += eps

        theta_minus = theta.copy()
        theta_minus[j] -= eps

        cost_plus = LinearRegressionGD.compute_cost(X, y, theta_plus)
        cost_minus = LinearRegressionGD.compute_cost(X, y, theta_minus)

        grad_num[j] = (cost_plus - cost_minus) / (2.0 * eps)

    return grad_num


def test_gradient_finite_difference_check():
    """Verify that analytic gradient matches numerical finite-difference gradient to ~1e-6."""
    np.random.seed(42)

    # Test across multiple random configurations
    configurations = [
        (50, 3),   # 50 samples, 3 features (+1 bias = 4 parameters)
        (200, 8),  # 200 samples, 8 features (+1 bias = 9 parameters)
        (30, 1),   # Simple 1-feature regression
    ]

    for m, n in configurations:
        X_raw = np.random.randn(m, n)
        X = add_bias(X_raw, position="first")
        y = np.random.randn(m) * 2.5 + 4.0

        # Test at several different theta points:
        theta_points = [
            np.zeros(n + 1),                      # Initialization point
            np.random.randn(n + 1),               # Arbitrary random point
            np.ones(n + 1) * 2.5,                 # Non-zero constant point
        ]

        for theta in theta_points:
            analytic_grad = LinearRegressionGD.compute_gradient(X, y, theta)
            numeric_grad = compute_numerical_gradient(X, y, theta, eps=1e-7)

            # Compute relative error norm
            diff_norm = np.linalg.norm(analytic_grad - numeric_grad)
            scale = np.linalg.norm(analytic_grad) + np.linalg.norm(numeric_grad)
            rel_error = diff_norm / max(scale, 1e-12)

            max_abs_diff = np.max(np.abs(analytic_grad - numeric_grad))

            # Both relative error and absolute error should be tightly within ~1e-6
            assert max_abs_diff < 1e-6, (
                f"Gradient check failed for shape ({m}, {n}): "
                f"max abs diff={max_abs_diff:.3e}, rel error={rel_error:.3e}"
            )

