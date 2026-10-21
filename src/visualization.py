"""
Module: src.visualization
Description: Generation of high-resolution statistical visualizations and model evaluation plots.
"""

from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent publication-quality styling
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 0.8


def plot_sales_distribution(
    df: pd.DataFrame,
    save_path: str | Path = "outputs/plots/01_sales_distribution.png"
) -> Path:
    """Plots histogram and KDE of Sales distribution with mean and median indicators."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    mean_val = df["Sales"].mean()
    median_val = df["Sales"].median()

    sns.histplot(
        df["Sales"],
        kde=True,
        color="#2b5c8f",
        bins=15,
        stat="density",
        ax=ax,
        edgecolor="white",
        linewidth=1.2,
        alpha=0.65
    )

    ax.axvline(mean_val, color="#d95f02", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}k")
    ax.axvline(median_val, color="#1b9e77", linestyle=":", linewidth=2, label=f"Median: {median_val:.2f}k")

    ax.set_title("Distribution of Sales Volume Across Markets", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Sales (Thousands of Units)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=300)
    plt.close(fig)
    return p


def plot_advertising_vs_sales(
    df: pd.DataFrame,
    save_path: str | Path = "outputs/plots/02_advertising_vs_sales_scatter.png"
) -> Path:
    """Creates a 3-panel scatter plot with regression trendlines for TV, Radio, and Newspaper vs Sales."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=300, sharey=True)
    channels = [("TV", "#2b5c8f"), ("Radio", "#d95f02"), ("Newspaper", "#1b9e77")]

    for i, (channel, color) in enumerate(channels):
        ax = axes[i]
        corr_val = df[channel].corr(df["Sales"])
        
        sns.regplot(
            data=df,
            x=channel,
            y="Sales",
            ax=ax,
            color=color,
            scatter_kws={"alpha": 0.65, "s": 40},
            line_kws={"linewidth": 2, "color": "#222222"}
        )
        ax.set_title(f"{channel} vs Sales (r = {corr_val:.3f})", fontsize=12, weight="bold")
        ax.set_xlabel(f"{channel} Advertising Spend ($1,000s)", fontsize=11)
        if i == 0:
            ax.set_ylabel("Sales (Thousands of Units)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Advertising Expenditure vs Product Sales by Channel", fontsize=15, weight="bold", y=1.03)
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_correlation_heatmap(
    df: pd.DataFrame,
    save_path: str | Path = "outputs/plots/03_correlation_heatmap.png"
) -> Path:
    """Plots an annotated correlation matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    numeric_df = df[["TV", "Radio", "Newspaper", "Sales"]]
    corr = numeric_df.corr()

    sns.heatmap(
        corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-0.2,
        vmax=1.0,
        square=True,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation (r)"},
        linewidths=1.0,
        linecolor="white",
        ax=ax
    )

    ax.set_title("Correlation Heatmap: Media Channels vs Sales", fontsize=13, weight="bold", pad=12)
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=300)
    plt.close(fig)
    return p


def plot_ad_spend_distribution(
    df: pd.DataFrame,
    save_path: str | Path = "outputs/plots/04_ad_spend_by_channel.png"
) -> Path:
    """Creates side-by-side boxplot and total share comparison of advertising channels."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
    channels = ["TV", "Radio", "Newspaper"]
    palette = {"TV": "#2b5c8f", "Radio": "#d95f02", "Newspaper": "#1b9e77"}

    # 1. Boxplot of expenditure per channel
    melted_df = df[channels].melt(var_name="Channel", value_name="Expenditure")
    sns.boxplot(
        data=melted_df,
        x="Channel",
        y="Expenditure",
        hue="Channel",
        palette=palette,
        legend=False,
        ax=ax1,
        width=0.45,
        boxprops=dict(alpha=0.8)
    )
    ax1.set_title("Budget Spread by Advertising Channel", fontsize=12, weight="bold")
    ax1.set_ylabel("Expenditure ($1,000s)", fontsize=11)
    ax1.set_xlabel("Media Channel", fontsize=11)

    # 2. Percentage of Total Budget Donut Chart
    total_spends = [df[c].sum() for c in channels]
    pie_colors = [palette[c] for c in channels]
    wedges, texts, autotexts = ax2.pie(
        total_spends,
        labels=channels,
        autopct="%1.1f%%",
        startangle=140,
        colors=pie_colors,
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2)
    )
    for at in autotexts:
        at.set_color("white")
        at.set_weight("bold")
    ax2.set_title("Overall Marketing Budget Share", fontsize=12, weight="bold")

    fig.suptitle("Comparative Analysis of Advertising Channels", fontsize=14, weight="bold", y=1.02)
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_actual_vs_predicted(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str = "Best Model",
    r2_score: Optional[float] = None,
    save_path: str | Path = "outputs/plots/05_actual_vs_predicted.png"
) -> Path:
    """Plots actual vs predicted sales values with ideal 45-degree reference line."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    min_val = min(y_true.min(), y_pred.min()) - 1
    max_val = max(y_true.max(), y_pred.max()) + 1

    ax.scatter(
        y_true,
        y_pred,
        color="#2b5c8f",
        alpha=0.75,
        edgecolors="w",
        s=60,
        label="Test Observations"
    )
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal Fit (y = x)")

    title = f"Actual vs Predicted Sales — {model_name}"
    if r2_score is not None:
        title += f" (R² = {r2_score:.4f})"

    ax.set_title(title, fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Actual Sales (Thousands of Units)", fontsize=11)
    ax.set_ylabel("Predicted Sales (Thousands of Units)", fontsize=11)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.legend(frameon=True, facecolor="white", edgecolor="#e0e0e0")

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=300)
    plt.close(fig)
    return p


def plot_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str = "Best Model",
    save_path: str | Path = "outputs/plots/06_residual_plot.png"
) -> Path:
    """Plots residuals vs predicted values and residual histogram for error validation."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

    # 1. Residuals vs Predicted (Homoscedasticity test)
    ax1.scatter(y_pred, residuals, color="#7570b3", alpha=0.75, edgecolors="w", s=50)
    ax1.axhline(0, color="red", linestyle="--", linewidth=1.8)
    ax1.set_title(f"Residuals vs Predicted — {model_name}", fontsize=12, weight="bold")
    ax1.set_xlabel("Predicted Sales (k units)", fontsize=11)
    ax1.set_ylabel("Residual Error (Actual - Predicted)", fontsize=11)

    # 2. Residual Distribution (Normality test)
    sns.histplot(residuals, kde=True, color="#7570b3", ax=ax2, edgecolor="white", bins=10)
    ax2.axvline(0, color="red", linestyle="--", linewidth=1.8)
    ax2.set_title("Residual Distribution (Error Normality)", fontsize=12, weight="bold")
    ax2.set_xlabel("Residual Error", fontsize=11)
    ax2.set_ylabel("Frequency", fontsize=11)

    fig.suptitle("Residual Analysis: Model Diagnostic Check", fontsize=14, weight="bold", y=1.02)
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p


def plot_feature_importance(
    importance_df: pd.DataFrame,
    save_path: str | Path = "outputs/plots/07_feature_importance.png"
) -> Path:
    """Plots feature importance comparison bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    sorted_df = importance_df.sort_values(by="Importance", ascending=False)

    sns.barplot(
        data=sorted_df,
        x="Importance",
        y="Feature",
        hue="Feature",
        palette="viridis",
        legend=False,
        ax=ax,
        edgecolor="white"
    )

    ax.set_title("Relative Feature Importance in Predicting Sales", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Relative Importance Score", fontsize=11)
    ax.set_ylabel("Advertising Feature", fontsize=11)

    for p_bar in ax.patches:
        width = p_bar.get_width()
        ax.annotate(
            f"{width:.3f}",
            (width, p_bar.get_y() + p_bar.get_height() / 2),
            ha="left",
            va="center",
            xytext=(5, 0),
            textcoords="offset points",
            fontweight="bold",
            fontsize=10
        )

    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(p, dpi=300)
    plt.close(fig)
    return p


if __name__ == "__main__":
    import sys
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.data_loader import load_raw_data
    df = load_raw_data()
    plot_sales_distribution(df, save_path=project_root / "outputs" / "plots" / "01_sales_distribution.png")
    print("Visualization self-test passed.")
