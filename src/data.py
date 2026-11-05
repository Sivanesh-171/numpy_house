"""Data loading, inspection, splitting, and imputation utilities.

This module provides generic functions to load tabular numerical datasets from CSV,
inspect schema and summary statistics, split into train and test partitions,
and impute missing values strictly using training set medians to avoid data leakage.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def download_california_housing_if_missing(csv_path: str | Path = "data/california_housing.csv") -> Path:
    """Download California Housing dataset using scikit-learn once and save to CSV.

    Per the project specification, sklearn is allowed ONLY to fetch the dataset once.
    Subsequent operations read strictly from the saved CSV.

    Parameters
    ----------
    csv_path : str or Path
        Destination path for the CSV file.

    Returns
    -------
    Path
        Path object pointing to the saved CSV.
    """
    path = Path(csv_path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[data.py] Fetching California Housing dataset via sklearn...")
        from sklearn.datasets import fetch_california_housing
        dataset = fetch_california_housing(as_frame=True)
        df = dataset.frame
        df.to_csv(path, index=False)
        print(f"[data.py] Successfully saved dataset to '{path}' ({len(df)} rows, {df.shape[1]} columns).")
    else:
        print(f"[data.py] Existing dataset found at '{path}'. Skipping download.")
    return path


def print_dataset_summary(df: pd.DataFrame, target_col: str) -> None:
    """Print an educational summary of the dataset: shape, dtypes, nulls, and statistics.

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset as a pandas DataFrame.
    target_col : str
        The name of the target column.
    """
    separator = "=" * 80
    print("\n" + separator)
    print("DATASET EXPLORATION & SUMMARY")
    print(separator)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target column: '{target_col}'")
    
    print("\n--- Column Data Types & Non-Null Counts ---")
    info_df = pd.DataFrame({
        "Dtype": df.dtypes,
        "Non-Null Count": df.notnull().sum(),
        "Null Count": df.isnull().sum(),
        "Null %": (df.isnull().sum() / len(df) * 100).round(2)
    })
    print(info_df.to_string())

    print("\n--- Descriptive Statistics (Pandas describe) ---")
    print(df.describe().T.to_string())
    print(separator + "\n")


def load_dataset(
    csv_path: str | Path,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Generic dataset loader with train/test splitting and median imputation.

    Educational Design & Anti-Leakage Protocol:
    1. Loads the CSV file into a pandas DataFrame.
    2. Separates feature matrix X and target vector y.
    3. Splits into train and test sets using the specified random seed.
    4. Computes the median of each feature strictly on the TRAIN partition.
    5. Imputes any missing values in both train and test using the TRAIN medians.
       This prevents information from the test distribution leaking into training.
    6. Returns clean NumPy arrays ready for matrix modeling.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.
    target_col : str
        Name of the numeric target column.
    test_size : float, default=0.2
        Proportion of the dataset allocated to the test split.
    random_state : int, default=42
        Seed for reproducible train/test partitioning.
    verbose : bool, default=True
        Whether to print dataset summary to console.

    Returns
    -------
    X_train : np.ndarray
        Training feature matrix of shape (n_train_samples, n_features).
    X_test : np.ndarray
        Testing feature matrix of shape (n_test_samples, n_features).
    y_train : np.ndarray
        Training target vector of shape (n_train_samples,).
    y_test : np.ndarray
        Testing target vector of shape (n_test_samples,).
    feature_names : list of str
        List containing the column names of features.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")

    # Read CSV strictly using Pandas
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame columns: {list(df.columns)}")

    if verbose:
        print_dataset_summary(df, target_col)

    # Separate features and target
    feature_cols = [c for c in df.columns if c != target_col]
    X_df = df[feature_cols].copy()
    y_df = df[target_col].copy()

    # Train / Test split
    X_train_df, X_test_df, y_train_series, y_test_series = train_test_split(
        X_df, y_df, test_size=test_size, random_state=random_state
    )

    # Missing value imputation:
    # CRITICAL: Compute medians on training data only to avoid data leakage!
    train_medians = X_train_df.median(numeric_only=True)
    
    # Impute missing values using train medians
    X_train_df = X_train_df.fillna(train_medians)
    X_test_df = X_test_df.fillna(train_medians)

    # If the target has nulls (rare in benchmarks), impute with train target median
    if y_train_series.isnull().any():
        train_y_median = y_train_series.median()
        y_train_series = y_train_series.fillna(train_y_median)
        y_test_series = y_test_series.fillna(train_y_median)

    # Convert to pure NumPy arrays (float64)
    X_train = X_train_df.to_numpy(dtype=np.float64)
    X_test = X_test_df.to_numpy(dtype=np.float64)
    y_train = y_train_series.to_numpy(dtype=np.float64)
    y_test = y_test_series.to_numpy(dtype=np.float64)

    return X_train, X_test, y_train, y_test, feature_cols

