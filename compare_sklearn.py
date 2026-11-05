"""Benchmark script: Scratch Linear Regression vs. Normal Equation vs. Scikit-Learn.

NOTE: Per project constraints, this is the ONLY file that imports sklearn.linear_model.
It verifies mathematical correctness by demonstrating that the batch gradient descent
and closed-form normal equation solvers converge to the exact same parameters and
test performance as scikit-learn's reference implementation.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.data import download_california_housing_if_missing, load_dataset
from src.preprocessing import StandardScaler, add_bias, unstandardize_coefficients
from src.model import LinearRegressionGD
from src.normal_equation import NormalEquationRegression
from src.metrics import mse, r2_score


def run_comparison(
    csv_path: str = "data/california_housing.csv",
    target_col: str = "MedHouseVal",
    alpha: float = 0.1,
    n_iters: int = 4000,
    tol: float | None = 1e-12,
    output_csv_path: str = "outputs/comparison_table.csv",
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Train scratch GD, Normal Equation, and sklearn on identical splits and compare.

    Parameters
    ----------
    csv_path : str
        Path to California Housing dataset CSV.
    target_col : str
        Target column name.
    alpha : float
        Learning rate for GD.
    n_iters : int
        Number of GD iterations.
    output_csv_path : str
        Where to save the comparison CSV.
    verbose : bool
        Whether to print comparison tables to stdout.

    Returns
    -------
    comparison_df : pd.DataFrame
        Side-by-side comparison of weights and metrics.
    diff_metrics : dict
        Max absolute differences between models.
    """
    # Ensure data is present
    download_california_housing_if_missing(csv_path)

    # 1. Load train and test splits (median imputed on train)
    X_train, X_test, y_train, y_test, feature_names = load_dataset(
        csv_path, target_col=target_col, test_size=0.2, random_state=42, verbose=False
    )

    # 2. Standardize features strictly fitting on train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Add bias column (column 0 = 1.0)
    X_train_bias = add_bias(X_train_scaled, position="first")
    X_test_bias = add_bias(X_test_scaled, position="first")

    # ==========================================
    # Model 1: Scratch Batch Gradient Descent
    # ==========================================
    gd_model = LinearRegressionGD(alpha=alpha, n_iters=n_iters, tol=tol, verbose=False)
    gd_model.fit(X_train_bias, y_train)

    train_pred_gd = gd_model.predict(X_train_bias)
    test_pred_gd = gd_model.predict(X_test_bias)

    gd_train_mse = mse(y_train, train_pred_gd)
    gd_test_mse = mse(y_test, test_pred_gd)
    gd_train_r2 = r2_score(y_train, train_pred_gd)
    gd_test_r2 = r2_score(y_test, test_pred_gd)

    # ==========================================
    # Model 2: Scratch Closed-Form Normal Equation
    # ==========================================
    ne_model = NormalEquationRegression()
    ne_model.fit(X_train_bias, y_train)

    train_pred_ne = ne_model.predict(X_train_bias)
    test_pred_ne = ne_model.predict(X_test_bias)

    ne_train_mse = mse(y_train, train_pred_ne)
    ne_test_mse = mse(y_test, test_pred_ne)
    ne_train_r2 = r2_score(y_train, train_pred_ne)
    ne_test_r2 = r2_score(y_test, test_pred_ne)

    # ==========================================
    # Model 3: Scikit-Learn Reference Model
    # ==========================================
    # Note: sklearn fits intercept internally on X_train_scaled
    sk_model = LinearRegression(fit_intercept=True)
    sk_model.fit(X_train_scaled, y_train)

    train_pred_sk = sk_model.predict(X_train_scaled)
    test_pred_sk = sk_model.predict(X_test_scaled)

    sk_train_mse = mse(y_train, train_pred_sk)
    sk_test_mse = mse(y_test, test_pred_sk)
    sk_train_r2 = r2_score(y_train, train_pred_sk)
    sk_test_r2 = r2_score(y_test, test_pred_sk)

    # Assemble parameter arrays: [intercept, coef_1, ..., coef_n]
    theta_gd = gd_model.theta_
    theta_ne = ne_model.theta_
    theta_sk = np.array([sk_model.intercept_, *sk_model.coef_], dtype=np.float64)

    # Parameter rows
    param_names = ["Intercept (theta_0)"] + [f"Coef ({f})" for f in feature_names]
    
    rows = []
    for name, w_gd, w_ne, w_sk in zip(param_names, theta_gd, theta_ne, theta_sk):
        rows.append({
            "Item": name,
            "Scratch GD": w_gd,
            "Normal Equation": w_ne,
            "Scikit-Learn": w_sk,
            "Abs Diff (GD vs SK)": abs(w_gd - w_sk),
            "Abs Diff (NE vs SK)": abs(w_ne - w_sk),
        })

    # Metric rows
    metrics_data = [
        ("Train MSE", gd_train_mse, ne_train_mse, sk_train_mse),
        ("Test MSE", gd_test_mse, ne_test_mse, sk_test_mse),
        ("Train R^2", gd_train_r2, ne_train_r2, sk_train_r2),
        ("Test R^2", gd_test_r2, ne_test_r2, sk_test_r2),
    ]

    for m_name, m_gd, m_ne, m_sk in metrics_data:
        rows.append({
            "Item": m_name,
            "Scratch GD": m_gd,
            "Normal Equation": m_ne,
            "Scikit-Learn": m_sk,
            "Abs Diff (GD vs SK)": abs(m_gd - m_sk),
            "Abs Diff (NE vs SK)": abs(m_ne - m_sk),
        })

    comparison_df = pd.DataFrame(rows)

    # Save to outputs directory
    out_path = Path(output_csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(out_path, index=False)

    max_coef_diff_gd = float(np.max(np.abs(theta_gd - theta_sk)))
    max_coef_diff_ne = float(np.max(np.abs(theta_ne - theta_sk)))
    test_r2_diff = abs(gd_test_r2 - sk_test_r2)
    test_mse_diff = abs(gd_test_mse - sk_test_mse)

    diff_summary = {
        "max_coef_diff_gd_vs_sk": max_coef_diff_gd,
        "max_coef_diff_ne_vs_sk": max_coef_diff_ne,
        "test_r2_diff": test_r2_diff,
        "test_mse_diff": test_mse_diff,
    }

    if verbose:
        header = "=" * 90
        print("\n" + header)
        print("VERIFICATION BENCHMARK: SCRATCH GD vs. NORMAL EQUATION vs. SCIKIT-LEARN")
        print(header)
        # Format floating numbers nicely for console display
        display_df = comparison_df.copy()
        for col in ["Scratch GD", "Normal Equation", "Scikit-Learn"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:12.7f}")
        for col in ["Abs Diff (GD vs SK)", "Abs Diff (NE vs SK)"]:
            display_df[col] = display_df[col].apply(lambda x: f"{x:12.2e}")

        print(display_df.to_string(index=False))
        print(header)
        print(f"Max Absolute Parameter Difference (Scratch GD vs Sklearn): {max_coef_diff_gd:.4e}")
        print(f"Max Absolute Parameter Difference (Normal Eq vs Sklearn):  {max_coef_diff_ne:.4e}")
        print(f"Test R^2 Difference (Scratch GD vs Sklearn):              {test_r2_diff:.4e}")
        print(f"Test MSE Difference (Scratch GD vs Sklearn):              {test_mse_diff:.4e}")
        print(f"Saved benchmark table to: '{out_path}'")
        print(header + "\n")

    return comparison_df, diff_summary


if __name__ == "__main__":
    run_comparison()
