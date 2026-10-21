"""
Module: src.data_loader
Description: Functions for locating, validating, and loading the sales prediction dataset.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd


def get_dataset_path(custom_path: Optional[str | Path] = None) -> Path:
    """
    Locates the dataset CSV file.
    
    Checks in the following order:
    1. User-supplied custom path (if provided)
    2. Local project folder 'data/sales_data.csv'
    3. Local project folder 'data/Advertising.csv'
    4. Downloads folder fallback
    """
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified dataset path does not exist: {p}")

    project_root = Path(__file__).resolve().parent.parent
    candidate_paths = [
        project_root / "data" / "sales_data.csv",
        project_root / "data" / "Advertising.csv",
        Path(r"C:\Users\vaish\Downloads\sales dataset\Advertising.csv"),
    ]

    for path in candidate_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Sales dataset not found! Please place 'sales_data.csv' or 'Advertising.csv' inside the 'data/' directory."
    )


def load_raw_data(file_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Loads the sales dataset from CSV into a pandas DataFrame.
    Automatically drops unnamed index columns and validates essential columns.
    
    Parameters:
        file_path: Optional custom path to dataset file.
        
    Returns:
        pd.DataFrame: Loaded raw dataset.
    """
    dataset_path = get_dataset_path(file_path)
    df = pd.read_csv(dataset_path)

    # Remove unnamed index column if present (e.g. 'Unnamed: 0' or ' ')
    unnamed_cols = [col for col in df.columns if "unnamed" in str(col).lower() or str(col).strip() == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Validate essential expected columns
    expected_cols = {"TV", "Radio", "Newspaper", "Sales"}
    available_cols = set(df.columns)
    missing_cols = expected_cols - available_cols
    if missing_cols:
        raise ValueError(
            f"Dataset at {dataset_path} is missing expected columns: {missing_cols}. Found: {list(df.columns)}"
        )

    return df


def get_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes an overview summary dictionary of the dataset.
    """
    return {
        "num_rows": int(df.shape[0]),
        "num_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }


if __name__ == "__main__":
    df = load_raw_data()
    summary = get_dataset_summary(df)
    print("Dataset Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nFirst 5 rows:\n", df.head())
