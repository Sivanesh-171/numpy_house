"""Tests verifying optimizer dynamics (monotonic cost decrease) and equivalence to scikit-learn."""

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from src.data import download_california_housing_if_missing, load_dataset
from src.preprocessing import StandardScaler, add_bias
from src.model import LinearRegressionGD, DivergenceWarning
from src.normal_equation import NormalEquationRegression
from src.metrics import mse, r2_score


@pytest.fixture(scope="module")
def california_splits():
    """Load and prepare California Housing splits once for the module tests."""
    csv_path = download_california_housing_if_missing("data/california_housing.csv")
    X_train, X_test, y_train, y_test, feature_names = load_dataset(
        csv_path, target_col="MedHouseVal", test_size=0.2, random_state=42, verbose=False
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_bias = add_bias(X_train_scaled, position="first")
    X_test_bias = add_bias(X_test_scaled, position="first")

    return {
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "X_train_bias": X_train_bias,
        "X_test_bias": X_test_bias,
        "y_train": y_train,
        "y_test": y_test,
    }


def test_cost_decreases_monotonically_for_good_alpha(california_splits):
    """Verify that the cost function J(theta) strictly decreases at every step for a stable alpha."""
    X_train_bias = california_splits["X_train_bias"]
    y_train = california_splits["y_train"]

    # Use alpha=0.1 on standardized features (stable, smooth convergence)
    model = LinearRegressionGD(alpha=0.1, n_iters=300, verbose=False)
    model.fit(X_train_bias, y_train)

    history = model.history_
    assert len(history) == 300

    # Verify monotonic decrease: J(t) <= J(t-1) for all t >= 1
    # Adding tiny epsilon (1e-12) for float precision near plateau
    for t in range(1, len(history)):
        assert history[t] <= history[t - 1] + 1e-12, (
            f"Cost increased at iteration {t}: prev={history[t-1]:.8f}, curr={history[t]:.8f}"
        )


def test_coefficients_match_sklearn_within_1e3(california_splits):
    """Verify that learned GD coefficients agree with scikit-learn's LinearRegression within 1e-3."""
    X_train_bias = california_splits["X_train_bias"]
    X_train_scaled = california_splits["X_train_scaled"]
    y_train = california_splits["y_train"]

    # Train scratch GD with sufficient iterations for full convergence
    gd_model = LinearRegressionGD(alpha=0.1, n_iters=3500, verbose=False)
    gd_model.fit(X_train_bias, y_train)

    # Train scikit-learn LinearRegression
    sk_model = LinearRegression(fit_intercept=True)
    sk_model.fit(X_train_scaled, y_train)

    sk_theta = np.array([sk_model.intercept_, *sk_model.coef_], dtype=np.float64)
    gd_theta = gd_model.theta_

    max_diff = np.max(np.abs(gd_theta - sk_theta))
    assert max_diff < 1e-3, f"Coefficients differ from sklearn by {max_diff:.4e} (threshold: 1e-3)"


def test_test_r2_matches_sklearn_within_1e3(california_splits):
    """Verify that test R^2 score of scratch GD agrees with scikit-learn within 1e-3."""
    X_train_bias = california_splits["X_train_bias"]
    X_test_bias = california_splits["X_test_bias"]
    X_train_scaled = california_splits["X_train_scaled"]
    X_test_scaled = california_splits["X_test_scaled"]
    y_train = california_splits["y_train"]
    y_test = california_splits["y_test"]

    gd_model = LinearRegressionGD(alpha=0.1, n_iters=3500, verbose=False)
    gd_model.fit(X_train_bias, y_train)
    y_pred_gd = gd_model.predict(X_test_bias)
    r2_gd = r2_score(y_test, y_pred_gd)

    sk_model = LinearRegression(fit_intercept=True)
    sk_model.fit(X_train_scaled, y_train)
    y_pred_sk = sk_model.predict(X_test_scaled)
    r2_sk = r2_score(y_test, y_pred_sk)

    r2_diff = abs(r2_gd - r2_sk)
    assert r2_diff < 1e-3, f"Test R^2 difference {r2_diff:.4e} exceeds 1e-3 threshold!"


def test_normal_equation_matches_sklearn_machine_precision(california_splits):
    """Verify closed-form normal equation matches scikit-learn to ~1e-12 precision."""
    X_train_bias = california_splits["X_train_bias"]
    X_train_scaled = california_splits["X_train_scaled"]
    y_train = california_splits["y_train"]

    ne_model = NormalEquationRegression()
    ne_model.fit(X_train_bias, y_train)

    sk_model = LinearRegression(fit_intercept=True)
    sk_model.fit(X_train_scaled, y_train)

    sk_theta = np.array([sk_model.intercept_, *sk_model.coef_], dtype=np.float64)
    ne_theta = ne_model.theta_

    max_diff = np.max(np.abs(ne_theta - sk_theta))
    assert max_diff < 1e-10, f"Normal equation differs from sklearn by {max_diff:.4e}"


def test_divergence_detection_on_excessive_alpha(california_splits):
    """Verify that an excessively large learning rate triggers DivergenceWarning or raises when configured."""
    X_train_bias = california_splits["X_train_bias"]
    y_train = california_splits["y_train"]

    # Alpha = 3.0 is well beyond the Lipschitz stability bound for this dataset
    gd_model_warning = LinearRegressionGD(alpha=3.0, n_iters=50, raise_on_divergence=False)
    with pytest.warns(DivergenceWarning):
        gd_model_warning.fit(X_train_bias, y_train)

    gd_model_strict = LinearRegressionGD(alpha=3.0, n_iters=50, raise_on_divergence=True)
    with pytest.raises(ValueError, match="diverged|increased"):
        gd_model_strict.fit(X_train_bias, y_train)

