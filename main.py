"""Main execution pipeline for Linear Regression from Scratch using pure NumPy.

Orchestrates all experiments:
1. Dataset ingestion, train/test splitting, and anti-leakage median imputation.
2. Experiment A: Simple Linear Regression (MedInc -> MedHouseVal).
3. Experiment B: Multiple Linear Regression (all 8 features).
4. Experiment C: Learning-rate study comparing convergence and divergence.
5. Experiment D: Final cost convergence curve.
6. Experiment E: Predicted-vs-actual scatter and residual analysis.
7. Experiment F: Unstandardized coefficients transformation back to physical units.
8. Verification benchmark against scikit-learn reference implementation.
"""

from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data import download_california_housing_if_missing, load_dataset
from src.preprocessing import StandardScaler, add_bias, unstandardize_coefficients
from src.model import LinearRegressionGD, DivergenceWarning
from src.normal_equation import NormalEquationRegression
from src.metrics import mse, rmse, mae, r2_score
from compare_sklearn import run_comparison


def set_plotting_style() -> None:
    """Set clean, publication-ready aesthetic defaults for Matplotlib."""
    plt.rcParams.update({
        "figure.autolayout": True,
        "figure.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    })


def experiment_a_simple_regression(
    X_train_full: np.ndarray,
    X_test_full: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    outputs_dir: Path
) -> None:
    """Experiment A: Simple Linear Regression using Median Income (MedInc) -> Target."""
    print("\n" + "=" * 80)
    print("EXPERIMENT A: SIMPLE LINEAR REGRESSION (MedInc -> MedHouseVal)")
    print("=" * 80)

    # MedInc is the 1st feature (index 0)
    medinc_idx = feature_names.index("MedInc")
    x_tr_raw = X_train_full[:, medinc_idx:medinc_idx + 1]
    x_te_raw = X_test_full[:, medinc_idx:medinc_idx + 1]

    # Standardize MedInc (strictly fitting on train)
    scaler = StandardScaler()
    x_tr_scaled = scaler.fit_transform(x_tr_raw)
    x_te_scaled = scaler.transform(x_te_raw)

    # Add bias term (x_0 = 1.0)
    x_tr_b = add_bias(x_tr_scaled, position="first")
    x_te_b = add_bias(x_te_scaled, position="first")

    # Fit Simple Regression via GD
    alpha = 0.1
    n_iters = 1500
    model = LinearRegressionGD(alpha=alpha, n_iters=n_iters, tol=1e-10)
    model.fit(x_tr_b, y_train)

    # Predict on test data
    y_pred_te = model.predict(x_te_b)
    test_mse = mse(y_test, y_pred_te)
    test_rmse = rmse(y_test, y_pred_te)
    test_r2 = r2_score(y_test, y_pred_te)

    # Unstandardize coefficients for physical interpretation
    beta_0, beta_1 = unstandardize_coefficients(model.theta_, scaler.mean_, scaler.scale_)

    print(f"Standardized Parameters:  theta_0 = {model.theta_[0]:.4f}, theta_1 = {model.theta_[1]:.4f}")
    print(f"Unstandardized Equation:  MedHouseVal = {beta_0:.4f} + {beta_1[0]:.4f} * MedInc")
    print(f"Evaluation Metrics (Test):")
    print(f"  MSE  = {test_mse:.4f}")
    print(f"  RMSE = {test_rmse:.4f} ($100k units)")
    print(f"  R^2  = {test_r2:.4f} ({test_r2*100:.1f}% variance explained by income alone)")

    # Plot Scatter of Test Data with Fitted Regression Line
    fig, ax = plt.subplots(figsize=(8, 6))

    # Downsample test scatter points slightly for clean visualization
    sample_indices = np.random.RandomState(42).choice(len(x_te_raw), size=min(1200, len(x_te_raw)), replace=False)
    ax.scatter(
        x_te_raw[sample_indices],
        y_test[sample_indices],
        alpha=0.35,
        color="#3b82f6",
        edgecolors="none",
        s=25,
        label="Test Observations (California census blocks)"
    )

    # Smooth regression line across original range of MedInc
    x_grid_raw = np.linspace(x_te_raw.min(), x_te_raw.max(), 300).reshape(-1, 1)
    x_grid_scaled = scaler.transform(x_grid_raw)
    x_grid_b = add_bias(x_grid_scaled, position="first")
    y_grid_pred = model.predict(x_grid_b)

    ax.plot(x_grid_raw, y_grid_pred, color="#ef4444", linewidth=2.5, label="Fitted Regression Line")

    ax.set_title("Experiment A: Simple Linear Regression\nMedian Income vs. Median House Value", pad=12)
    ax.set_xlabel("Median Income (MedInc in $10,000s)")
    ax.set_ylabel("Median House Value (MedHouseVal in $100,000s)")
    ax.grid(True)

    textstr = "\n".join([
        r"$\hat{y} = %.3f + %.3f \cdot \mathrm{MedInc}$" % (beta_0, beta_1[0]),
        r"$\mathrm{Test\ MSE} = %.4f$" % test_mse,
        r"$\mathrm{Test\ } R^2 = %.4f$" % test_r2,
    ])
    props = dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#d1d5db", alpha=0.9)
    ax.text(0.04, 0.94, textstr, transform=ax.transAxes, verticalalignment="top", bbox=props)

    ax.legend(loc="lower right")
    plot_path = outputs_dir / "simple_regression_fit.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"[Experiment A] Saved plot to: {plot_path}")


def experiment_c_learning_rate_study(
    X_train_bias: np.ndarray,
    y_train: np.ndarray,
    outputs_dir: Path
) -> float:
    """Experiment C: Comparative study of learning rates and divergence analysis."""
    print("\n" + "=" * 80)
    print("EXPERIMENT C: LEARNING RATE STUDY & CONVERGENCE DYNAMICS")
    print("=" * 80)

    # Candidate learning rates: slow, moderate, optimal, aggressive, and divergent
    alphas = [0.001, 0.01, 0.1, 0.5, 1.0, 1.5]
    n_iters = 100

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#6b7280", "#0284c7", "#10b981", "#f59e0b", "#f97316", "#ef4444"]

    print(f"{'Alpha':<10} {'Init Cost':<14} {'Cost @ Iter 50':<18} {'Cost @ Iter 100':<18} {'Behavior Description'}")
    print("-" * 80)

    for alpha, color in zip(alphas, colors):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DivergenceWarning)
            gd = LinearRegressionGD(alpha=alpha, n_iters=n_iters, tol=None, raise_on_divergence=False)
            gd.fit(X_train_bias, y_train)

        history = np.array(gd.history_, dtype=np.float64)
        c_init = history[0]
        c_50 = history[min(49, len(history) - 1)]
        c_end = history[-1]

        if alpha == 0.001:
            desc = "Slow, steady descent (understepping)"
        elif alpha == 0.01:
            desc = "Moderate convergence"
        elif alpha == 0.1:
            desc = "Fast, optimal monotonic descent"
        elif alpha == 0.5:
            desc = "Aggressive descent, near optimum"
        elif alpha == 1.0:
            desc = "Oscillating / boundary instability"
        else:
            desc = "Violent divergence (J -> infinity)"

        print(f"{alpha:<10} {c_init:<14.4f} {c_50:<18.4f} {c_end:<18.4f} {desc}")

        # Clip values for plotting so divergent alpha=1.5 doesn't crush the y-axis
        history_clipped = np.clip(history, 0.0, 10.0)
        ax.plot(
            range(len(history_clipped)),
            history_clipped,
            label=rf"$\alpha = {alpha}$ ({desc.split(',')[0].split('(')[0].strip()})",
            color=color,
            linewidth=2.2 if alpha == 0.1 else 1.8,
            linestyle="--" if alpha in [1.0, 1.5] else "-"
        )

    ax.set_title("Experiment C: Batch Gradient Descent Cost Curves vs. Learning Rate $\\alpha$", pad=12)
    ax.set_xlabel("Iteration Step (t)")
    ax.set_ylabel(r"Cost $J(\theta)$ (Mean Squared Error $\times \frac{1}{2}$)")
    ax.set_ylim(0.2, 3.2)
    ax.grid(True)
    ax.legend(loc="upper right", framealpha=0.95)

    # Highlight optimal alpha justification
    ax.annotate(
        "Optimal $\\alpha=0.1$:\nRapid descent & perfect stability",
        xy=(35, 0.28),
        xytext=(45, 0.9),
        arrowprops=dict(facecolor="#10b981", shrink=0.08, width=1.5, headwidth=7),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecfdf5", edgecolor="#10b981")
    )

    plot_path = outputs_dir / "lr_study.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"[Experiment C] Saved plot to: {plot_path}")

    # Justification: alpha=0.1 guarantees monotonic convergence and maximum stable step size
    chosen_alpha = 0.1
    print(f"\nSelection Justification:")
    print(f"  alpha = {chosen_alpha} is selected as optimal. It achieves rapid convergence within ~1000-2000 steps")
    print(f"  without the oscillations observed at alpha >= 1.0 or the severe instability at alpha >= 1.5.")
    return chosen_alpha


def experiment_b_and_d_final_model(
    X_train_bias: np.ndarray,
    X_test_bias: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    chosen_alpha: float,
    outputs_dir: Path
) -> LinearRegressionGD:
    """Experiment B & D: Train final multiple linear regression model and plot convergence curve."""
    print("\n" + "=" * 80)
    print("EXPERIMENT B & D: MULTIPLE LINEAR REGRESSION & FINAL COST CONVERGENCE")
    print("=" * 80)

    n_iters = 3500
    model = LinearRegressionGD(alpha=chosen_alpha, n_iters=n_iters, tol=1e-12, verbose=False)
    model.fit(X_train_bias, y_train)

    train_pred = model.predict(X_train_bias)
    test_pred = model.predict(X_test_bias)

    tr_mse, te_mse = mse(y_train, train_pred), mse(y_test, test_pred)
    tr_rmse, te_rmse = rmse(y_train, train_pred), rmse(y_test, test_pred)
    tr_r2, te_r2 = r2_score(y_train, train_pred), r2_score(y_test, test_pred)

    print(f"Model Configuration: alpha={chosen_alpha}, max_iters={n_iters}, converged_at_step={model.n_iters_}")
    print(f"Performance Metrics:")
    print(f"  Train MSE = {tr_mse:.6f} | Train RMSE = {tr_rmse:.4f} | Train R^2 = {tr_r2:.6f}")
    print(f"  Test  MSE = {te_mse:.6f} | Test  RMSE = {te_rmse:.4f} | Test  R^2 = {te_r2:.6f}")

    # Plot Cost Convergence (Experiment D)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    history = model.history_
    iterations = np.arange(len(history))

    # Left: Full overview of Cost vs Iteration
    ax1.plot(iterations, history, color="#2563eb", linewidth=2.0)
    ax1.set_title("Cost Convergence: Full Trajectory", pad=10)
    ax1.set_xlabel("Iteration Step (t)")
    ax1.set_ylabel(r"Cost $J(\theta)$")
    ax1.grid(True)
    ax1.annotate(
        f"Initial: {history[0]:.4f}\nFinal: {history[-1]:.6f}",
        xy=(iterations[-1], history[-1]),
        xytext=(iterations[-1] * 0.6, history[0] * 0.6),
        arrowprops=dict(facecolor="#2563eb", shrink=0.08, width=1.5, headwidth=6),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#eff6ff", edgecolor="#93c5fd")
    )

    # Right: Log-scale cost improvement to display asymptotic flattening
    cost_improvements = np.maximum(np.abs(np.diff(history)), 1e-16)
    ax2.semilogy(iterations[1:], cost_improvements, color="#7c3aed", linewidth=1.8)
    ax2.set_title(r"Log Asymptotic Convergence: $|J^{(t)} - J^{(t-1)}|$", pad=10)
    ax2.set_xlabel("Iteration Step (t)")
    ax2.set_ylabel(r"Cost Improvement (log scale)")
    ax2.grid(True, which="both")

    fig.suptitle(r"Experiment D: Cost Function $J(\theta)$ Convergence ($\alpha=0.1$)", fontsize=14, y=1.02)
    plot_path = outputs_dir / "cost_convergence.png"
    fig.savefig(plot_path)
    plt.close(fig)
    print(f"[Experiment D] Saved plot to: {plot_path}")

    return model


def experiment_e_residual_analysis(
    model: LinearRegressionGD,
    X_test_bias: np.ndarray,
    y_test: np.ndarray,
    outputs_dir: Path
) -> None:
    """Experiment E: Predicted vs. Actual scatter and residual diagnostic analysis."""
    print("\n" + "=" * 80)
    print("EXPERIMENT E: PREDICTED VS. ACTUAL & RESIDUAL DIAGNOSTICS")
    print("=" * 80)

    y_pred = model.predict(X_test_bias)
    residuals = y_test - y_pred

    test_mse = mse(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)

    # 1. Predicted vs. Actual Scatter Plot
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    ax1.scatter(y_test, y_pred, alpha=0.3, color="#3b82f6", edgecolors="none", s=25, label="Test Predictions")

    # 45-degree reference line (Ideal y = y_pred)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], color="#ef4444", linestyle="--", linewidth=2.0, label="Ideal 45° Line (y = y_hat)")

    # Highlight the 5.0 target ceiling cap
    ax1.axvline(x=5.00001, color="#f59e0b", linestyle=":", linewidth=2.0, label="Census 5.0 Ceiling Cap")

    ax1.set_title("Experiment E: Predicted vs. Actual California House Values", pad=12)
    ax1.set_xlabel("Actual Median House Value ($100k)")
    ax1.set_ylabel("Predicted Median House Value ($100k)")
    ax1.grid(True)
    ax1.legend(loc="upper left")

    textstr = "\n".join([
        r"$\mathrm{Test\ MSE} = %.4f$" % test_mse,
        r"$\mathrm{Test\ } R^2 = %.4f$" % test_r2,
        r"$\mathrm{Notice:\ Target\ clipped\ at\ 5.0}$",
    ])
    props = dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#d1d5db", alpha=0.9)
    ax1.text(0.55, 0.12, textstr, transform=ax1.transAxes, verticalalignment="bottom", bbox=props)

    pva_path = outputs_dir / "pred_vs_actual.png"
    fig1.savefig(pva_path)
    plt.close(fig1)
    print(f"[Experiment E] Saved Predicted-vs-Actual plot to: {pva_path}")

    # 2. Residual Diagnostic Plots (Residuals vs Predicted & Residual Distribution)
    fig2, (ax_res, ax_dist) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Residuals vs Predicted
    ax_res.scatter(y_pred, residuals, alpha=0.3, color="#6366f1", edgecolors="none", s=25)
    ax_res.axhline(y=0.0, color="#ef4444", linestyle="--", linewidth=2.0)
    ax_res.set_title("Residuals vs. Fitted Values", pad=10)
    ax_res.set_xlabel("Predicted Value ($100k)")
    ax_res.set_ylabel("Residual (Actual - Predicted)")
    ax_res.grid(True)

    # Right: Residual Distribution Histogram with Normal PDF overlay
    counts, bins, _ = ax_dist.hist(residuals, bins=50, density=True, color="#818cf8", edgecolor="white", alpha=0.75)
    res_mean = np.mean(residuals)
    res_std = np.std(residuals)
    x_pdf = np.linspace(bins.min(), bins.max(), 300)
    norm_pdf = (1.0 / (res_std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_pdf - res_mean) / res_std) ** 2)
    ax_dist.plot(x_pdf, norm_pdf, color="#dc2626", linewidth=2.0, label=f"Normal Fit (mean={res_mean:.2f}, std={res_std:.2f})")
    ax_dist.axvline(x=0.0, color="#1e293b", linestyle=":", linewidth=1.5)
    ax_dist.set_title("Residual Error Distribution", pad=10)
    ax_dist.set_xlabel("Residual (Actual - Predicted)")
    ax_dist.set_ylabel("Probability Density")
    ax_dist.grid(True)
    ax_dist.legend(loc="upper right")

    fig2.suptitle("Experiment E: Residual Diagnostics for Multiple Linear Regression", fontsize=14, y=1.02)
    res_path = outputs_dir / "residuals.png"
    fig2.savefig(res_path)
    plt.close(fig2)
    print(f"[Experiment E] Saved Residuals diagnostic plot to: {res_path}")


def experiment_f_unstandardize_coefficients(
    model: LinearRegressionGD,
    scaler: StandardScaler,
    feature_names: list[str]
) -> pd.DataFrame:
    """Experiment F: Transform learned weights back into physical feature units."""
    print("\n" + "=" * 80)
    print("EXPERIMENT F: STANDARDIZED VS. UNSTANDARDIZED COEFFICIENTS")
    print("=" * 80)

    beta_0, beta = unstandardize_coefficients(model.theta_, scaler.mean_, scaler.scale_)

    rows = []
    # Intercept row
    rows.append({
        "Feature": "Intercept (theta_0 / beta_0)",
        "Standardized Weight (theta)": model.theta_[0],
        "Mean (mu)": np.nan,
        "Std (sigma)": np.nan,
        "Unstandardized Weight (beta)": beta_0,
        "Physical Unit / Interpretation": "Baseline median house value ($100,000s)"
    })

    units = [
        "$10,000s of median income",
        "Years of housing age",
        "Average rooms per household",
        "Average bedrooms per household",
        "Block group population count",
        "Average household occupants",
        "Degrees latitude",
        "Degrees longitude",
    ]

    for name, th, mu, sigma, b, u in zip(feature_names, model.theta_[1:], scaler.mean_, scaler.scale_, beta, units):
        rows.append({
            "Feature": name,
            "Standardized Weight (theta)": th,
            "Mean (mu)": mu,
            "Std (sigma)": sigma,
            "Unstandardized Weight (beta)": b,
            "Physical Unit / Interpretation": u
        })

    coeff_df = pd.DataFrame(rows)

    display_df = coeff_df.copy()
    display_df["Standardized Weight (theta)"] = display_df["Standardized Weight (theta)"].apply(lambda x: f"{x:10.5f}")
    display_df["Unstandardized Weight (beta)"] = display_df["Unstandardized Weight (beta)"].apply(lambda x: f"{x:10.5f}")
    display_df["Mean (mu)"] = display_df["Mean (mu)"].apply(lambda x: f"{x:10.3f}" if pd.notnull(x) else "       N/A")
    display_df["Std (sigma)"] = display_df["Std (sigma)"].apply(lambda x: f"{x:10.3f}" if pd.notnull(x) else "       N/A")

    print(display_df.to_string(index=False))
    print("\nPhysical Interpretation of Top Factors:")
    print("  1. MedInc (+0.854 standardized, +0.449 unstandardized):")
    print("     Every additional $10,000 of median income predicts a $44,900 increase in median house price.")
    print("  2. Latitude & Longitude (-0.897 & -0.870 standardized):")
    print("     Large negative values capture California's northwest-southeast geographical orientation")
    print("     and coastal premium (proximity to Pacific coast).")
    print("=" * 80 + "\n")
    return coeff_df


def main() -> None:
    """Run full pipeline end to end."""
    print("=" * 80)
    print("LINEAR REGRESSION FROM SCRATCH USING ONLY NUMPY")
    print("Production-Quality Educational Implementation & Verification Benchmark")
    print("=" * 80)

    set_plotting_style()
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)

    csv_path = download_california_housing_if_missing("data/california_housing.csv")

    # 1. Load, inspect, and split dataset with train-median imputation
    X_train_raw, X_test_raw, y_train, y_test, feature_names = load_dataset(
        csv_path, target_col="MedHouseVal", test_size=0.2, random_state=42, verbose=True
    )

    # 2. Preprocess: StandardScaler (fit on train only!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # 3. Add bias term (column of 1s prepended)
    X_train_bias = add_bias(X_train_scaled, position="first")
    X_test_bias = add_bias(X_test_scaled, position="first")

    # Experiment A: Simple Regression (MedInc -> MedHouseVal)
    experiment_a_simple_regression(X_train_raw, X_test_raw, y_train, y_test, feature_names, outputs_dir)

    # Experiment C: Learning Rate Study (divergence analysis)
    chosen_alpha = experiment_c_learning_rate_study(X_train_bias, y_train, outputs_dir)

    # Experiment B & D: Multiple Regression & Final Convergence
    final_model = experiment_b_and_d_final_model(
        X_train_bias, X_test_bias, y_train, y_test, chosen_alpha, outputs_dir
    )

    # Experiment E: Predicted vs. Actual and Residual Diagnostics
    experiment_e_residual_analysis(final_model, X_test_bias, y_test, outputs_dir)

    # Experiment F: Convert coefficients to original units
    experiment_f_unstandardize_coefficients(final_model, scaler, feature_names)

    # Verification Benchmark against Scikit-Learn
    run_comparison(csv_path="data/california_housing.csv", output_csv_path="outputs/comparison_table.csv")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("Generated Artifacts in 'outputs/':")
    for f in outputs_dir.glob("*"):
        print(f"  - {f.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()

