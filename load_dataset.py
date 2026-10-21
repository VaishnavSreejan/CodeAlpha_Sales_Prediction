"""
Module: load_dataset.py
Purpose: Load and inspect the Advertising Sales dataset for sales prediction modeling.
"""

import os
from pathlib import Path
import pandas as pd


def get_dataset_path() -> Path:
    """Finds the dataset path, checking project directory first, then Downloads."""
    local_sales = Path(__file__).resolve().parent / "data" / "sales_data.csv"
    local_adv = Path(__file__).resolve().parent / "data" / "Advertising.csv"
    downloads_path = Path(r"C:\Users\vaish\Downloads\sales dataset\Advertising.csv")
    
    if local_sales.exists():
        return local_sales
    elif local_adv.exists():
        return local_adv
    elif downloads_path.exists():
        return downloads_path
    else:
        raise FileNotFoundError(
            f"Dataset not found at {local_sales}, {local_adv}, or {downloads_path}"
        )


def load_data(file_path: str | Path | None = None) -> pd.DataFrame:
    """
    Loads the sales advertising dataset into a pandas DataFrame.
    
    Parameters:
        file_path (str | Path, optional): Custom path to the dataset CSV file.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame with features (TV, Radio, Newspaper) and target (Sales).
    """
    path = Path(file_path) if file_path else get_dataset_path()
    
    # Read CSV. The first column is often an unnamed index column.
    df = pd.read_csv(path)
    
    # If the first column is unnamed or just an index, drop it
    unnamed_cols = [c for c in df.columns if "Unnamed" in str(c) or c == ""]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)
        
    return df


def inspect_data(df: pd.DataFrame) -> None:
    """Prints comprehensive summary and inspection of the dataset."""
    print("=" * 60)
    print("                 DATASET OVERVIEW")
    print("=" * 60)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns\n")
    
    print("--- First 5 Rows ---")
    print(df.head())
    print("\n--- Last 5 Rows ---")
    print(df.tail())
    
    print("\n--- Data Types & Non-Null Counts ---")
    print(df.info())
    
    print("\n--- Missing Values ---")
    null_counts = df.isnull().sum()
    print(null_counts)
    
    print("\n--- Duplicate Rows ---")
    duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {duplicates}")
    
    print("\n--- Summary Statistics ---")
    print(df.describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]])
    
    if "Sales" in df.columns:
        print("\n--- Correlation with Sales Target ---")
        corr = df.corr()
        print(corr["Sales"].sort_values(ascending=False))
        
    print("=" * 60)


if __name__ == "__main__":
    dataset_path = get_dataset_path()
    print(f"Loading dataset from: {dataset_path}")
    df = load_data(dataset_path)
    inspect_data(df)
