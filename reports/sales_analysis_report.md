# CodeAlpha Task 4: Sales Prediction Analysis Report

**Project Name:** Sales Prediction using Python  
**Internship Provider:** CodeAlpha  
**Domain:** Data Science & Machine Learning  
**Target Variable:** `Sales` (in thousands of units)  
**Input Features:** `TV`, `Radio`, `Newspaper` (advertising spend in $1,000s)  

---

## 1. Executive Summary

This project investigates the relationship between advertising expenditures across multiple media channels (Television, Radio, and Newspaper) and sales performance. Using empirical machine learning models and statistical diagnostics, we quantify channel impact, model sales trajectories, and extract data-driven marketing guidance.

### Key Headline Metrics:
- **Total Campaigns Analyzed:** 200 market regions
- **Mean Product Sales:** 14.02k units (Std: 5.22k)
- **Top Performing Model:** **Gradient Boosting**
- **Best Test $R^2$ Score:** 0.9831 (98.31% of sales variance explained)
- **Best Test RMSE:** 0.7295 thousand units
- **Best Test MAE:** 0.6181 thousand units

---

## 2. Channel Spend & Return Analysis

| Channel | Mean Spend ($k) | Budget Share (%) | Correlation with Sales ($r$) |
| :--- | :---: | :---: | :---: |
| **TV** | $147.04k | 73.2% | `0.7822` |
| **Radio** | $23.26k | 11.6% | `0.5762` |
| **Newspaper** | $30.55k | 15.2% | `0.2283` |

### Strategic Insights:
1. **TV Dominance:** TV advertising shows the highest correlation with sales volume ($r = 0.7822$) and represents the largest proportion of total marketing expenditure (73.2%).
2. **Radio as a High-Efficiency Multiplier:** Radio demonstrates strong linear correlation ($r = 0.5762$) while requiring significantly lower absolute investment.
3. **Newspaper Redundancy:** Newspaper advertising displays low direct correlation ($r = 0.2283$) and marginal feature importance in multivariable models.

---

## 3. Model Comparison & Benchmarks

| Model Name | Train $R^2$ | Test $R^2$ | Test RMSE | Test MAE | Test MSE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | 0.8957 | 0.8994 | 1.7816 | 1.4608 | 3.1741 |
| **Decision Tree** | 0.9838 | 0.9145 | 1.6424 | 1.1944 | 2.6974 |
| **Random Forest** | 0.9946 | 0.9801 | 0.7935 | 0.6319 | 0.6297 |
| **Gradient Boosting** **(Best)** | 0.9987 | 0.9831 | 0.7295 | 0.6181 | 0.5322 |

---

## 4. Feature Importance

| Feature | Importance Score | Relative Share (%) |
| :--- | :---: | :---: |
| **TV** | 0.6126 | 61.3% |
| **Radio** | 0.3809 | 38.1% |
| **Newspaper** | 0.0065 | 0.7% |

---

## 5. Marketing Recommendations

1. **Reallocate Print Budget:** Shift budget from Newspaper to combined TV and Radio campaigns.
2. **Leverage Media Synergy:** Coordinate Radio spots alongside major TV ad flights to maximize brand recall.
3. **Predictive Budgeting:** Utilize the serialized pipeline (`models/sales_prediction_model.joblib`) or Streamlit application (`app.py`) for data-driven campaign budget planning.
