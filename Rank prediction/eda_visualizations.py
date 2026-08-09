"""
Phase 1.3 — EDA Visualizations
===============================
Generates comprehensive visualizations of the JEE Advanced dataset.

Input:  interpolated_data.csv
Output: plots/ directory with all charts
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(__file__)
INPUT_FILE = os.path.join(BASE_DIR, "interpolated_data.csv")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Aesthetic defaults
plt.rcParams.update({
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "text.color": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "grid.color": "#21262d",
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

PALETTE = sns.color_palette("Spectral", 17)


def plot_marks_vs_rank(df):
    """Overlay plot of Marks vs Rank for each year — shows exponential decay."""
    fig, ax = plt.subplots(figsize=(14, 8))
    years = sorted(df["Year"].unique())
    for i, year in enumerate(years):
        ydf = df[df["Year"] == year].sort_values("AIR")
        ax.plot(ydf["AIR"], ydf["Mark"], label=str(year),
                color=PALETTE[i % len(PALETTE)], alpha=0.85, linewidth=1.3)
    ax.set_xlabel("All India Rank (AIR)")
    ax.set_ylabel("Marks")
    ax.set_title("Marks vs AIR — All Years (Exponential Decay)")
    ax.legend(fontsize=7, ncol=3, loc="upper right",
              framealpha=0.3, edgecolor="#30363d")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "01_marks_vs_rank_all_years.png"))
    plt.close()
    print("  [OK] Marks vs Rank (all years)")


def plot_percentage_vs_rank(df):
    """Normalized percentage vs rank — reveals difficulty variation."""
    fig, ax = plt.subplots(figsize=(14, 8))
    years = sorted(df["Year"].unique())
    for i, year in enumerate(years):
        ydf = df[df["Year"] == year].sort_values("AIR")
        ax.plot(ydf["AIR"], ydf["Percentage"], label=str(year),
                color=PALETTE[i % len(PALETTE)], alpha=0.85, linewidth=1.3)
    ax.set_xlabel("All India Rank (AIR)")
    ax.set_ylabel("Percentage Score (%)")
    ax.set_title("Percentage Score vs AIR — Normalized by Max Marks")
    ax.legend(fontsize=7, ncol=3, loc="upper right",
              framealpha=0.3, edgecolor="#30363d")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "02_percentage_vs_rank_all_years.png"))
    plt.close()
    print("  [OK] Percentage vs Rank (all years)")


def plot_marks_at_fixed_ranks(df):
    """Distribution of marks at key rank milestones across years."""
    milestones = [1, 101, 1001, 5001, 10001, 20001]
    data = []
    for rank in milestones:
        for year in sorted(df["Year"].unique()):
            row = df[(df["Year"] == year) & (df["AIR"] == rank)]
            if len(row) > 0 and row.iloc[0]["Mark"] == row.iloc[0]["Mark"]:
                data.append({
                    "Rank": f"AIR {rank}",
                    "Year": year,
                    "Percentage": row.iloc[0]["Percentage"]
                })
    plot_df = pd.DataFrame(data)
    if plot_df.empty:
        print("  [!] No data for fixed rank milestones")
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.boxplot(data=plot_df, x="Rank", y="Percentage", ax=ax,
                palette="coolwarm", linewidth=0.8,
                flierprops=dict(marker="o", markersize=4))
    ax.set_title("Percentage Score Distribution at Key Ranks (Across All Years)")
    ax.set_ylabel("Percentage Score (%)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "03_score_distribution_at_milestones.png"))
    plt.close()
    print("  [OK] Score distribution at rank milestones")


def plot_data_availability_heatmap(df):
    """Heatmap of data availability: year × rank range."""
    years = sorted(df["Year"].unique())
    # Bin ranks
    bins = list(range(0, 35001, 1000))
    labels = [f"{b//1000}K-{(b+1000)//1000}K" for b in bins[:-1]]

    heatmap_data = []
    for year in years:
        ydf = df[(df["Year"] == year) & (df["Mark"].notna())]
        counts, _ = np.histogram(ydf["AIR"], bins=bins)
        heatmap_data.append(counts)

    heatmap_df = pd.DataFrame(heatmap_data, index=years, columns=labels)

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(heatmap_df, annot=False, cmap="YlGnBu", ax=ax,
                linewidths=0.3, linecolor="#30363d",
                cbar_kws={"label": "# data points"})
    ax.set_title("Data Availability Heatmap (Year × Rank Range)")
    ax.set_ylabel("Year")
    ax.set_xlabel("Rank Range")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "04_data_availability_heatmap.png"))
    plt.close()
    print("  [OK] Data availability heatmap")


def plot_difficulty_trend(df):
    """Marks at rank 1000 across years — proxy for difficulty."""
    data = []
    for year in sorted(df["Year"].unique()):
        row = df[(df["Year"] == year) & (df["AIR"] == 1001)]
        if len(row) > 0 and pd.notna(row.iloc[0]["Percentage"]):
            data.append({"Year": year, "Percentage": row.iloc[0]["Percentage"]})
    if not data:
        return
    plot_df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(plot_df)))
    bars = ax.bar(plot_df["Year"].astype(str), plot_df["Percentage"],
                  color=colors, edgecolor="#30363d", linewidth=0.5)
    ax.set_title("Exam Difficulty Proxy — % Score Needed for AIR 1001")
    ax.set_ylabel("Percentage Score (%)")
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.3, axis="y")
    # Add value labels
    for bar, val in zip(bars, plot_df["Percentage"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=7,
                color="#c9d1d9")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "05_difficulty_trend.png"))
    plt.close()
    print("  [OK] Difficulty trend (marks at rank 1000)")


def plot_log_rank_vs_percentage(df):
    """Log(Rank) vs Percentage — shows the linearization effect."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Left: raw
    for i, year in enumerate(sorted(df["Year"].unique())):
        ydf = df[df["Year"] == year].sort_values("AIR")
        axes[0].plot(ydf["Percentage"], ydf["AIR"],
                     color=PALETTE[i % len(PALETTE)], alpha=0.6, linewidth=1)
    axes[0].set_xlabel("Percentage Score (%)")
    axes[0].set_ylabel("AIR (Rank)")
    axes[0].set_title("Raw: Percentage vs Rank")
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_xaxis()

    # Right: log-transformed
    for i, year in enumerate(sorted(df["Year"].unique())):
        ydf = df[df["Year"] == year].sort_values("AIR")
        axes[1].plot(ydf["Percentage"], np.log(ydf["AIR"]),
                     color=PALETTE[i % len(PALETTE)], alpha=0.6, linewidth=1)
    axes[1].set_xlabel("Percentage Score (%)")
    axes[1].set_ylabel("log(AIR)")
    axes[1].set_title("Log-Transformed: Percentage vs log(Rank)")
    axes[1].grid(True, alpha=0.3)
    axes[1].invert_xaxis()

    plt.suptitle("Effect of Log Transformation on Rank",
                 fontsize=14, color="#58a6ff", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "06_log_transform_comparison.png"),
                bbox_inches="tight")
    plt.close()
    print("  [OK] Log-transform comparison")


def main():
    print("=" * 60)
    print("Phase 1.3: EDA Visualizations")
    print("=" * 60)

    df = pd.read_csv(INPUT_FILE)
    print(f"\nLoaded {len(df)} records")
    print(f"Generating plots in: {PLOTS_DIR}\n")

    plot_marks_vs_rank(df)
    plot_percentage_vs_rank(df)
    plot_marks_at_fixed_ranks(df)
    plot_data_availability_heatmap(df)
    plot_difficulty_trend(df)
    plot_log_rank_vs_percentage(df)

    print(f"\n[OK] All plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
