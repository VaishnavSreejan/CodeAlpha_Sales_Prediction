"""
Streamlit Web Application: Sales Prediction using Python
CodeAlpha Task 4 — Interactive Machine Learning Application
"""

from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Sales Prediction AI | CodeAlpha",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2563eb;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
    }
    .info-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 4px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to load model
@st.cache_resource
def get_model():
    model_path = Path(__file__).resolve().parent / "models" / "sales_prediction_model.joblib"
    if not model_path.exists():
        return None
    from src.prediction import load_trained_model
    return load_trained_model(model_path)


# Title & Header
st.markdown('<div class="main-header">📈 Sales Prediction using Python</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">CodeAlpha Data Science Internship — Task 4 | Predictive Advertising Intelligence</div>', unsafe_allow_html=True)

pipeline = get_model()
if pipeline is None:
    st.error("⚠️ Trained model not found! Please run `python main.py` in your terminal to train and serialize the model pipeline first.")
    st.stop()

# Sidebar: Project Info & Parameters
st.sidebar.title("🛠️ Campaign Controls")
st.sidebar.markdown("Configure your advertising expenditure across media channels to forecast product sales volume.")

unit_price = st.sidebar.number_input(
    "Estimated Unit Selling Price ($)",
    min_value=1.0,
    max_value=1000.0,
    value=25.0,
    step=1.0,
    help="Used to estimate total sales revenue alongside unit volume."
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 About the Model")
st.sidebar.markdown("""
- **Algorithm:** Gradient Boosting Regressor Pipeline
- **R² Score:** 0.9831 (98.3% variance explained)
- **RMSE:** 0.7295 k units
- **Input Features:** TV, Radio, Newspaper
- **Dataset:** 200 Multi-market campaigns
""")

# Main Content Layout: 2 Columns (Inputs & Forecast)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("1. Enter Advertising Budget ($1,000s)")
    st.caption("Enter the planned expenditure for each channel in thousands of dollars (e.g., $50k = 50.0).")

    tv_spend = st.slider(
        "📺 TV Advertising Spend ($1,000s):",
        min_value=0.0,
        max_value=350.0,
        value=150.0,
        step=1.0,
        help="Historical range: $0.7k - $296.4k. TV is the strongest driver of baseline sales."
    )

    radio_spend = st.slider(
        "📻 Radio Advertising Spend ($1,000s):",
        min_value=0.0,
        max_value=60.0,
        value=25.0,
        step=0.5,
        help="Historical range: $0.0k - $49.6k. Radio creates high synergy with TV."
    )

    newspaper_spend = st.slider(
        "📰 Newspaper Advertising Spend ($1,000s):",
        min_value=0.0,
        max_value=120.0,
        value=15.0,
        step=1.0,
        help="Historical range: $0.3k - $114.0k. Modest direct correlation with sales."
    )

    total_spend = tv_spend + radio_spend + newspaper_spend
    st.markdown("#### 💰 Budget Allocation Breakdown")
    
    if total_spend > 0:
        budget_df = pd.DataFrame({
            "Channel": ["TV", "Radio", "Newspaper"],
            "Spend ($k)": [tv_spend, radio_spend, newspaper_spend],
            "Share (%)": [
                round((tv_spend / total_spend) * 100, 1),
                round((radio_spend / total_spend) * 100, 1),
                round((newspaper_spend / total_spend) * 100, 1),
            ]
        })
        st.dataframe(budget_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Total budget is currently $0.0k. Please allocate spend to generate predictions.")

    predict_btn = st.button("🚀 Forecast Sales Volume", type="primary", use_container_width=True)

with col2:
    st.subheader("2. Sales Forecast & Projections")
    
    # Run prediction automatically or on button click
    from src.prediction import predict_sales
    input_df = pd.DataFrame([{
        "TV": tv_spend,
        "Radio": radio_spend,
        "Newspaper": newspaper_spend
    }])
    
    pred_k_units = predict_sales(input_df)[0]
    total_units = int(pred_k_units * 1000)
    est_revenue = total_units * unit_price
    roi_ratio = (est_revenue / (total_spend * 1000)) if total_spend > 0 else 0.0

    # Display Metrics in Columns
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Predicted Sales Volume</div>
            <div class="metric-value">{pred_k_units:.2f}k</div>
            <div style="color: #6b7280; font-size: 0.85rem;">≈ {total_units:,} units</div>
        </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Estimated Gross Revenue</div>
            <div class="metric-value" style="color: #059669;">${est_revenue:,.0f}</div>
            <div style="color: #6b7280; font-size: 0.85rem;">@ ${unit_price}/unit</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Contextual Comparison
    historical_mean = 14.02  # from dataset
    diff_pct = ((pred_k_units - historical_mean) / historical_mean) * 100
    
    if diff_pct >= 0:
        st.success(f"📈 Forecasted sales are **{diff_pct:.1f}% above** the historical market benchmark average ({historical_mean}k units).")
    else:
        st.info(f"📉 Forecasted sales are **{abs(diff_pct):.1f}% below** the historical market benchmark average ({historical_mean}k units).")

    # Marketing Recommendation based on channel shares
    st.markdown("### 💡 Strategic Channel Guidance")
    if newspaper_spend > (0.35 * total_spend) and total_spend > 0:
        st.warning("⚠️ High Print Allocation: Newspaper constitutes over 35% of your budget. Statistical analysis indicates that reallocating print spend into TV or Radio yields higher sales lift.")
    elif tv_spend >= (0.6 * total_spend) and radio_spend >= (0.2 * total_spend):
        st.success("✅ Balanced High-Performance Mix: Strong TV baseline combined with active Radio presence maximizes multi-channel synergy.")
    else:
        st.info("ℹ️ Optimal Mix: TV provides scale, while Radio drives cost-effective incremental sales.")

    # Educational Disclaimer
    st.markdown("""
    <div class="info-box">
        <strong>⚠️ Estimation Disclaimer:</strong><br>
        Machine learning predictions reflect patterns observed in the 200 historical market campaigns. 
        Actual real-world performance may be influenced by seasonality, competitor promotions, creative quality, and macro-economic factors.
    </div>
    """, unsafe_allow_html=True)

# Bottom section: Visual Insights Tabs
st.markdown("---")
st.subheader("3. Exploratory Data Analysis & Diagnostic Insights")

tab1, tab2, tab3 = st.tabs(["📊 Market Distributions", "🔍 Correlation & Channel Spread", "🎯 Model Diagnostics"])

plots_dir = Path(__file__).resolve().parent / "outputs" / "plots"

with tab1:
    c1, c2 = st.columns(2)
    if (plots_dir / "01_sales_distribution.png").exists():
        c1.image(str(plots_dir / "01_sales_distribution.png"), caption="Sales Volume Distribution across Markets", use_container_width=True)
    if (plots_dir / "02_advertising_vs_sales_scatter.png").exists():
        c2.image(str(plots_dir / "02_advertising_vs_sales_scatter.png"), caption="Advertising Spend vs Sales by Channel", use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    if (plots_dir / "03_correlation_heatmap.png").exists():
        c1.image(str(plots_dir / "03_correlation_heatmap.png"), caption="Pearson Correlation Matrix", use_container_width=True)
    if (plots_dir / "04_ad_spend_by_channel.png").exists():
        c2.image(str(plots_dir / "04_ad_spend_by_channel.png"), caption="Channel Expenditure Comparison", use_container_width=True)

with tab3:
    c1, c2 = st.columns(2)
    if (plots_dir / "05_actual_vs_predicted.png").exists():
        c1.image(str(plots_dir / "05_actual_vs_predicted.png"), caption="Actual vs Predicted Sales on Test Split", use_container_width=True)
    if (plots_dir / "07_feature_importance.png").exists():
        c2.image(str(plots_dir / "07_feature_importance.png"), caption="Relative Feature Importance", use_container_width=True)

st.markdown("---")
st.caption("CodeAlpha Data Science Internship Task 4 | Developed by Vaishnav Sreejan | Powered by Scikit-Learn & Streamlit")
