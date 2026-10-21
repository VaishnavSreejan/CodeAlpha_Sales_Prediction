"""
Module: src.model_training
Description: Training, comparison, evaluation, and serialization of regression pipelines.
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns a dictionary of candidate regressor pipelines to compare.
    Each candidate is wrapped in a standard scaling pipeline.
    """
    return {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression())
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", DecisionTreeRegressor(max_depth=5, min_samples_split=4, random_state=random_state))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", RandomForestRegressor(n_estimators=100, max_depth=6, random_state=random_state))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state))
        ]),
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculates MAE, MSE, RMSE, and R2 score.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": round(float(mae), 4),
        "MSE": round(float(mse), 4),
        "RMSE": round(float(rmse), 4),
        "R2": round(float(r2), 4),
    }


def train_and_compare_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    random_state: int = 42
) -> Tuple[Dict[str, Dict[str, float]], str, Pipeline, Dict[str, np.ndarray]]:
    """
    Trains all candidate models on the training set, evaluates them on the test set,
    and identifies the best performing model based on test R2 score and RMSE.
    
    Returns:
        metrics_dict: Dictionary with metrics for all models (train and test)
        best_name: Name of the top-performing model
        best_pipeline: Fitted pipeline of the top model
        test_predictions: Dict mapping model name to test predictions array
    """
    candidate_models = get_candidate_models(random_state=random_state)
    all_metrics: Dict[str, Dict[str, float]] = {}
    test_predictions: Dict[str, np.ndarray] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}

    best_name = ""
    best_score = -float("inf")

    for name, pipeline in candidate_models.items():
        # Fit on training data ONLY
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predictions
        train_preds = pipeline.predict(X_train)
        test_preds = pipeline.predict(X_test)
        test_predictions[name] = test_preds

        train_metrics = evaluate_predictions(y_train.values, train_preds)
        test_metrics = evaluate_predictions(y_test.values, test_preds)

        combined_metrics = {
            "Train_R2": train_metrics["R2"],
            "Train_RMSE": train_metrics["RMSE"],
            "R2": test_metrics["R2"],
            "MAE": test_metrics["MAE"],
            "MSE": test_metrics["MSE"],
            "RMSE": test_metrics["RMSE"],
        }
        all_metrics[name] = combined_metrics

        # Select model with highest test R2 score
        if test_metrics["R2"] > best_score:
            best_score = test_metrics["R2"]
            best_name = name

    best_pipeline = fitted_pipelines[best_name]
    return all_metrics, best_name, best_pipeline, test_predictions


def extract_feature_importance(
    pipeline: Pipeline,
    feature_names: List[str],
    model_name: str
) -> pd.DataFrame:
    """
    Extracts feature importances or standardized coefficients from the fitted pipeline.
    """
    regressor = pipeline.named_steps["regressor"]
    
    if hasattr(regressor, "feature_importances_"):
        scores = regressor.feature_importances_
        metric_name = "Tree Importance (Gini / MDI)"
    elif hasattr(regressor, "coef_"):
        # For linear models, use absolute magnitude of standardized coefficients
        scores = np.abs(regressor.coef_)
        metric_name = "Abs Standardized Coefficient"
    else:
        scores = np.ones(len(feature_names)) / len(feature_names)
        metric_name = "Uniform Weight"

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": scores,
        "Metric": metric_name,
        "Model": model_name
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Normalize to sum to 1.0 for comparability
    total = importance_df["Importance"].sum()
    if total > 0:
        importance_df["Relative_Importance_Pct"] = (importance_df["Importance"] / total * 100).round(2)
    else:
        importance_df["Relative_Importance_Pct"] = (100 / len(feature_names)).round(2)

    return importance_df


def save_model(pipeline: Pipeline, file_path: str | Path = "models/sales_prediction_model.joblib") -> Path:
    """Serializes the fitted pipeline to disk using joblib."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, p)
    return p


def save_metrics(metrics: Dict[str, Any], file_path: str | Path = "outputs/model_metrics.json") -> Path:
    """Saves model evaluation metrics dictionary to JSON."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    return p


def save_feature_importance(df: pd.DataFrame, file_path: str | Path = "outputs/feature_importance.csv") -> Path:
    """Saves feature importance DataFrame to CSV."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return p


if __name__ == "__main__":
    import sys
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data_loader import load_raw_data
    from src.preprocessing import clean_dataset, split_data

    raw_df = load_raw_data()
    clean_df = clean_dataset(raw_df)
    feature_cols = ["TV", "Radio", "Newspaper"]
    X_tr, X_te, y_tr, y_te = split_data(clean_df, feature_cols=feature_cols)

    metrics, best_name, best_pipe, preds = train_and_compare_models(X_tr, X_te, y_tr, y_te)
    print("Model Evaluation Metrics:")
    for m_name, m_vals in metrics.items():
        print(f"  {m_name:20s}: R2={m_vals['R2']:.4f}, RMSE={m_vals['RMSE']:.4f}, MAE={m_vals['MAE']:.4f}")
    print(f"\nBest Model: {best_name}")

    imp_df = extract_feature_importance(best_pipe, feature_cols, best_name)
    print("\nFeature Importance:\n", imp_df)
