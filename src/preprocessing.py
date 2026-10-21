"""
Module: src.preprocessing
Description: Data cleaning, validation, feature engineering, and train-test splitting.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def clean_dataset(df: pd.DataFrame, save_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Cleans the raw dataset:
    - Drops any residual index columns
    - Handles missing values (verifies or imputes with column median)
    - Checks for and removes exact duplicate records
    - Ensures non-negative numeric constraints on spend and sales
    - Optionally saves cleaned dataset to disk
    """
    cleaned = df.copy()

    # Drop any remaining unnamed columns
    unnamed = [c for c in cleaned.columns if "unnamed" in str(c).lower() or str(c).strip() == ""]
    if unnamed:
        cleaned = cleaned.drop(columns=unnamed)

    # Missing value handling
    initial_nulls = cleaned.isnull().sum().sum()
    if initial_nulls > 0:
        for col in cleaned.select_dtypes(include=[np.number]).columns:
            if cleaned[col].isnull().any():
                median_val = cleaned[col].median()
                cleaned[col] = cleaned[col].fillna(median_val)

    # Duplicate check and removal
    num_duplicates = cleaned.duplicated().sum()
    if num_duplicates > 0:
        cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    # Sanity check: non-negative values for advertising spend and sales
    numeric_cols = ["TV", "Radio", "Newspaper", "Sales"]
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            # Cap any negative erroneous values at 0
            if (cleaned[col] < 0).any():
                cleaned[col] = cleaned[col].clip(lower=0.0)

    # Save cleaned dataset if path requested
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_csv(p, index=False)

    return cleaned


def add_engineered_features(df: pd.DataFrame, include_synergy: bool = True) -> pd.DataFrame:
    """
    Creates meaningful engineered features from advertising spending:
    - Total_Spend: Combined advertising budget across all channels
    - TV_Share: Proportion of total budget spent on TV
    - Radio_Share: Proportion of total budget spent on Radio
    - Newspaper_Share: Proportion of total budget spent on Newspaper
    - TV_Radio_Interaction: Marketing synergy interaction between TV and Radio campaigns
    """
    data = df.copy()
    if all(col in data.columns for col in ["TV", "Radio", "Newspaper"]):
        data["Total_Spend"] = data["TV"] + data["Radio"] + data["Newspaper"]
        safe_total = data["Total_Spend"].replace(0, 1e-6)
        data["TV_Share"] = (data["TV"] / safe_total).round(4)
        data["Radio_Share"] = (data["Radio"] / safe_total).round(4)
        data["Newspaper_Share"] = (data["Newspaper"] / safe_total).round(4)

        if include_synergy:
            # Interaction synergy term (TV x Radio)
            data["TV_Radio_Interaction"] = data["TV"] * data["Radio"]

    return data


def split_data(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = "Sales",
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits features and target into train and test sets.
    
    Parameters:
        df: Input DataFrame
        feature_cols: List of column names to use as features (default: all except target)
        target_col: Target variable name (default: 'Sales')
        test_size: Proportion of test split (default: 0.2 for 80/20 split)
        random_state: Reproducible random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import sys
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data_loader import load_raw_data

    raw_df = load_raw_data()
    clean_df = clean_dataset(raw_df, save_path=project_root / "outputs" / "cleaned_sales_data.csv")
    print(f"Cleaned data shape: {clean_df.shape}")
    
    eng_df = add_engineered_features(clean_df)
    print(f"Engineered features:\n{eng_df.head(3)}")
    
    X_tr, X_te, y_tr, y_te = split_data(clean_df, feature_cols=["TV", "Radio", "Newspaper"])
    print(f"Train set: {X_tr.shape}, Test set: {X_te.shape}")
