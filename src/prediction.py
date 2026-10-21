"""
Module: src.prediction
Description: Production-ready inference module for reloading the model pipeline
and generating validated sales predictions.
"""

from pathlib import Path
from typing import Dict, List, Union, Any
import pandas as pd
import numpy as np
import joblib


def load_trained_model(model_path: str | Path | None = None) -> Any:
    """
    Loads the trained scikit-learn pipeline from disk.
    """
    if model_path is None:
        project_root = Path(__file__).resolve().parent.parent
        model_path = project_root / "models" / "sales_prediction_model.joblib"
    else:
        model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. Please run 'python main.py' to train and save the model."
        )

    pipeline = joblib.load(model_path)
    return pipeline


def validate_input_data(data: Union[Dict[str, float], List[Dict[str, float]], pd.DataFrame]) -> pd.DataFrame:
    """
    Validates and normalizes input data into a standardized pandas DataFrame.
    
    Checks:
    - Required feature columns: 'TV', 'Radio', 'Newspaper'
    - Numeric data types
    - Non-negative constraints on advertising expenditure
    """
    required_cols = ["TV", "Radio", "Newspaper"]

    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(
            f"Unsupported input type '{type(data)}'. Expected dict, list of dicts, or pandas DataFrame."
        )

    # Check for missing required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Input is missing required advertising channel columns: {missing_cols}. Provided columns: {list(df.columns)}"
        )

    # Ensure consistent column order
    df = df[required_cols]

    # Validate numeric types and check for NaN
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            raise ValueError(f"Column '{col}' contains non-numeric or missing (NaN) values.")
        if (df[col] < 0).any():
            raise ValueError(f"Advertising expenditure for '{col}' cannot be negative.")

    return df


def predict_sales(
    input_data: Union[Dict[str, float], List[Dict[str, float]], pd.DataFrame],
    model_path: str | Path | None = None
) -> np.ndarray:
    """
    Generates predicted sales (in thousands of units) using the fitted pipeline.
    
    Parameters:
        input_data: Dict, list of dicts, or DataFrame containing 'TV', 'Radio', 'Newspaper' values.
        model_path: Optional path to serialized model file.
        
    Returns:
        np.ndarray: Predicted sales volume (in thousands of units).
    """
    validated_df = validate_input_data(input_data)
    pipeline = load_trained_model(model_path)
    predictions = pipeline.predict(validated_df)
    
    # Clip any negative model artifacts to 0 (sales cannot be negative)
    predictions = np.clip(predictions, a_min=0.0, a_max=None)
    return np.round(predictions, 3)


if __name__ == "__main__":
    # Self-test sample input
    sample_campaign = {"TV": 230.1, "Radio": 37.8, "Newspaper": 69.2}
    try:
        pred = predict_sales(sample_campaign)
        print(f"Sample Campaign: {sample_campaign}")
        print(f"Predicted Sales: {pred[0]} thousand units")
    except Exception as e:
        print(f"Prediction test info: {e}")
