"""
Module: main.py
Description: Main execution pipeline for CodeAlpha Task 4 — Sales Prediction using Python.
Executes data ingestion, validation, preprocessing, EDA, model training & comparison,
evaluation, visualization, feature importance extraction, and pipeline serialization.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_raw_data, get_dataset_summary
from src.preprocessing import clean_dataset, split_data
from src.analysis import (
    compute_statistical_summary,
    compute_correlations,
    compute_channel_metrics,
    generate_key_findings,
    save_key_findings,
)
from src.visualization import (
    plot_sales_distribution,
    plot_advertising_vs_sales,
    plot_correlation_heatmap,
    plot_ad_spend_distribution,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importance,
)
from src.model_training import (
    train_and_compare_models,
    extract_feature_importance,
    save_model,
    save_metrics,
    save_feature_importance,
)
from src.prediction import predict_sales


def generate_markdown_report(
    clean_df: pd.DataFrame,
    metrics_dict: dict,
    best_model_name: str,
    importance_df: pd.DataFrame,
    report_path: str | Path = "reports/sales_analysis_report.md"
) -> Path:
    """Creates a comprehensive markdown business and technical report."""
    channel_df = compute_channel_metrics(clean_df)
    corrs = compute_correlations(clean_df, target_col="Sales")

    content = f"""# CodeAlpha Task 4: Sales Prediction Analysis Report

**Project Name:** Sales Prediction using Python  
**Internship Provider:** CodeAlpha  
**Domain:** Data Science & Machine Learning  
**Target Variable:** `Sales` (in thousands of units)  
**Input Features:** `TV`, `Radio`, `Newspaper` (advertising spend in $1,000s)  

---

## 1. Executive Summary

This project investigates the relationship between advertising expenditures across multiple media channels (Television, Radio, and Newspaper) and sales performance. Using empirical machine learning models and statistical diagnostics, we quantify channel impact, model sales trajectories, and extract data-driven marketing guidance.

### Key Headline Metrics:
- **Total Campaigns Analyzed:** {len(clean_df)} market regions
- **Mean Product Sales:** {clean_df['Sales'].mean():.2f}k units (Std: {clean_df['Sales'].std():.2f}k)
- **Top Performing Model:** **{best_model_name}**
- **Best Test $R^2$ Score:** {metrics_dict[best_model_name]['R2']:.4f} ({metrics_dict[best_model_name]['R2']*100:.2f}% of sales variance explained)
- **Best Test RMSE:** {metrics_dict[best_model_name]['RMSE']:.4f} thousand units
- **Best Test MAE:** {metrics_dict[best_model_name]['MAE']:.4f} thousand units

---

## 2. Channel Spend & Return Analysis

| Channel | Mean Spend ($k) | Budget Share (%) | Correlation with Sales ($r$) |
| :--- | :---: | :---: | :---: |
| **TV** | ${channel_df.loc['TV', 'Mean_Spend ($k)']:.2f}k | {channel_df.loc['TV', 'Budget_Share (%)']:.1f}% | `{corrs.get('TV', 0.0):.4f}` |
| **Radio** | ${channel_df.loc['Radio', 'Mean_Spend ($k)']:.2f}k | {channel_df.loc['Radio', 'Budget_Share (%)']:.1f}% | `{corrs.get('Radio', 0.0):.4f}` |
| **Newspaper** | ${channel_df.loc['Newspaper', 'Mean_Spend ($k)']:.2f}k | {channel_df.loc['Newspaper', 'Budget_Share (%)']:.1f}% | `{corrs.get('Newspaper', 0.0):.4f}` |

### Strategic Insights:
1. **TV Dominance:** TV advertising shows the highest correlation with sales volume ($r = {corrs.get('TV', 0.0):.4f}$) and represents the largest proportion of total marketing expenditure ({channel_df.loc['TV', 'Budget_Share (%)']:.1f}%).
2. **Radio as a High-Efficiency Multiplier:** Radio demonstrates strong linear correlation ($r = {corrs.get('Radio', 0.0):.4f}$) while requiring significantly lower absolute investment.
3. **Newspaper Redundancy:** Newspaper advertising displays low direct correlation ($r = {corrs.get('Newspaper', 0.0):.4f}$) and marginal feature importance in multivariable models.

---

## 3. Model Comparison & Benchmarks

| Model Name | Train $R^2$ | Test $R^2$ | Test RMSE | Test MAE | Test MSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""

    for name, m in metrics_dict.items():
        is_best = " **(Best)**" if name == best_model_name else ""
        content += f"| **{name}**{is_best} | {m['Train_R2']:.4f} | {m['R2']:.4f} | {m['RMSE']:.4f} | {m['MAE']:.4f} | {m['MSE']:.4f} |\n"

    content += f"""
---

## 4. Feature Importance

| Feature | Importance Score | Relative Share (%) |
| :--- | :---: | :---: |
"""
    for _, row in importance_df.iterrows():
        content += f"| **{row['Feature']}** | {row['Importance']:.4f} | {row['Relative_Importance_Pct']:.1f}% |\n"

    content += f"""
---

## 5. Marketing Recommendations

1. **Reallocate Print Budget:** Shift budget from Newspaper to combined TV and Radio campaigns.
2. **Leverage Media Synergy:** Coordinate Radio spots alongside major TV ad flights to maximize brand recall.
3. **Predictive Budgeting:** Utilize the serialized pipeline (`models/sales_prediction_model.joblib`) or Streamlit application (`app.py`) for data-driven campaign budget planning.
"""

    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def run_pipeline() -> None:
    """Executes the complete machine learning project workflow."""
    print("=" * 75)
    print("       CODEALPHA TASK 4: SALES PREDICTION USING PYTHON")
    print("=" * 75)

    # 1. Load Data
    print("\n[Step 1/8] Loading dataset...")
    raw_df = load_raw_data()
    summary = get_dataset_summary(raw_df)
    print(f"Loaded {summary['num_rows']} rows and {summary['num_columns']} columns.")
    print(f"Columns: {summary['columns']}")
    print(f"Missing values: {summary['total_missing']} | Duplicates: {summary['duplicate_rows']}")

    # 2. Clean & Preprocess
    print("\n[Step 2/8] Cleaning and validating dataset...")
    cleaned_df = clean_dataset(raw_df, save_path=PROJECT_ROOT / "outputs" / "cleaned_sales_data.csv")
    print(f"Cleaned dataset saved to: outputs/cleaned_sales_data.csv")

    # 3. Statistical Analysis & Key Findings
    print("\n[Step 3/8] Computing statistical insights & channel metrics...")
    stats_df = compute_statistical_summary(cleaned_df)
    channel_df = compute_channel_metrics(cleaned_df)
    correlations = compute_correlations(cleaned_df, target_col="Sales")

    # 4. Exploratory Visualizations
    print("\n[Step 4/8] Generating exploratory data visualization plots...")
    p1 = plot_sales_distribution(cleaned_df)
    p2 = plot_advertising_vs_sales(cleaned_df)
    p3 = plot_correlation_heatmap(cleaned_df)
    p4 = plot_ad_spend_distribution(cleaned_df)
    print(f"  - Saved: {p1.name}")
    print(f"  - Saved: {p2.name}")
    print(f"  - Saved: {p3.name}")
    print(f"  - Saved: {p4.name}")

    # 5. Train & Compare Models
    print("\n[Step 5/8] Splitting data (80/20) and comparing candidate regression models...")
    feature_cols = ["TV", "Radio", "Newspaper"]
    X_train, X_test, y_train, y_test = split_data(
        cleaned_df, feature_cols=feature_cols, target_col="Sales", test_size=0.2, random_state=42
    )
    print(f"  Training set size: {X_train.shape[0]} samples")
    print(f"  Testing set size:  {X_test.shape[0]} samples")

    metrics_dict, best_name, best_pipeline, test_preds = train_and_compare_models(
        X_train, X_test, y_train, y_test, random_state=42
    )

    print("\n" + "-" * 75)
    print(f"{'Model':22s} | {'Train R2':8s} | {'Test R2':8s} | {'Test RMSE':9s} | {'Test MAE':8s}")
    print("-" * 75)
    for m_name, m_vals in metrics_dict.items():
        flag = " [BEST]" if m_name == best_name else ""
        print(
            f"{m_name:22s} | {m_vals['Train_R2']:8.4f} | {m_vals['R2']:8.4f} | {m_vals['RMSE']:9.4f} | {m_vals['MAE']:8.4f}{flag}"
        )
    print("-" * 75)
    print(f"\nTop Performing Model Selected: {best_name}")

    # 6. Save Model & Metrics
    print("\n[Step 6/8] Serializing best model pipeline and evaluation metrics...")
    model_path = save_model(best_pipeline)
    metrics_path = save_metrics(metrics_dict)
    print(f"  - Saved Model Pipeline: {model_path}")
    print(f"  - Saved Model Metrics:  {metrics_path}")

    # 7. Feature Importance & Diagnostic Plots
    print("\n[Step 7/8] Analyzing feature importance & generating evaluation plots...")
    importance_df = extract_feature_importance(best_pipeline, feature_cols, best_name)
    save_feature_importance(importance_df)
    
    p5 = plot_actual_vs_predicted(
        y_test, test_preds[best_name], model_name=best_name, r2_score=metrics_dict[best_name]["R2"]
    )
    p6 = plot_residuals(y_test, test_preds[best_name], model_name=best_name)
    p7 = plot_feature_importance(importance_df)
    print(f"  - Saved: {p5.name}")
    print(f"  - Saved: {p6.name}")
    print(f"  - Saved: {p7.name}")
    print(f"  - Saved: outputs/feature_importance.csv")

    # 8. Reports & Inference Verification
    print("\n[Step 8/8] Generating business reports & verifying model inference...")
    findings_text = generate_key_findings(cleaned_df, metrics_dict)
    save_key_findings(findings_text)
    report_file = generate_markdown_report(cleaned_df, metrics_dict, best_name, importance_df)
    print(f"  - Saved: outputs/key_findings.txt")
    print(f"  - Saved: {report_file}")

    # Test sample inference using reload function
    sample_input = {"TV": 230.1, "Radio": 37.8, "Newspaper": 69.2}
    pred_val = predict_sales(sample_input)
    print(f"\nInference Self-Test:")
    print(f"  Input Campaign: {sample_input}")
    print(f"  Predicted Sales: {pred_val[0]} thousand units")

    print("\n" + "=" * 75)
    print("Execution complete! All project assets, plots, and models generated successfully.")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
