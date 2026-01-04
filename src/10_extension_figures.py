"""
Extension Figures: Publication-Ready Visualizations

This script generates figures for both extensions:

EXTENSION 1 - LEARNING OVER TIME:
- Figure 1: Time series of similarity measures (rolling, cumulative, decay)
- Figure 2: Market response evolution over time
- Figure 3: Crisis vs non-crisis regime comparison

EXTENSION 2 - RISK VS UNCERTAINTY:
- Figure 4: Risk and Uncertainty indices over time
- Figure 5: Scatter plot of risk vs uncertainty with market response
- Figure 6: Decomposed sentiment components

COMBINED:
- Figure 7: Coefficient comparison across specifications
- Figure 8: Model fit comparison (R² across models)

All figures are publication-ready with clear labels and saved as PDF/PNG.
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# --- Force working directory to project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)

# --- Paths ---
DATA_PATH = Path("data/processed/regression_dataset_extended.csv")
FIGURES_DIR = Path("outputs/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# --- Load data ---
if not DATA_PATH.exists():
    DATA_PATH = Path("data/processed/event_study_constant_mean.csv")
    
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])

print("=" * 80)
print("GENERATING EXTENSION FIGURES")
print("=" * 80)
print(f"Loaded {len(df)} observations")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Output directory: {FIGURES_DIR.resolve()}")

# Define crisis periods for shading
crisis_periods = [
    ("2008-09-15", "2009-06-30", "GFC"),
    ("2010-05-01", "2012-12-31", "Eurozone"),
    ("2020-03-01", "2021-06-30", "COVID"),
    ("2022-02-24", "2022-12-31", "Ukraine"),
]

def add_crisis_shading(ax, crisis_periods, alpha=0.15):
    """Add shaded regions for crisis periods."""
    for start, end, label in crisis_periods:
        ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                   alpha=alpha, color='red', label=f'{label}' if label == 'GFC' else '')

# =============================================================================
# FIGURE 1: SIMILARITY MEASURES OVER TIME
# =============================================================================
print("\n[1/8] Generating Figure 1: Similarity Measures...")

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Panel A: Jaccard similarity
ax1 = axes[0]
ax1.plot(df["date"], df["similarity_jaccard"], 'b-', linewidth=1.5, alpha=0.8, label="Jaccard (t vs t-1)")
if "similarity_rolling5" in df.columns:
    ax1.plot(df["date"], df["similarity_rolling5"], 'r--', linewidth=1.5, alpha=0.8, label="Rolling (5-stmt avg)")
add_crisis_shading(ax1, crisis_periods)
ax1.set_ylabel("Similarity")
ax1.set_title("A. Communication Similarity Measures")
ax1.legend(loc="upper right")
ax1.set_ylim(0, 1)

# Panel B: Decay-weighted similarity
ax2 = axes[1]
if "similarity_decay" in df.columns:
    ax2.plot(df["date"], df["similarity_decay"], 'g-', linewidth=1.5, alpha=0.8, label="Decay-weighted (λ=0.5)")
if "cumulative_similarity" in df.columns:
    ax2.plot(df["date"], df["cumulative_similarity"], 'm--', linewidth=1.5, alpha=0.8, label="Cumulative average")
add_crisis_shading(ax2, crisis_periods)
ax2.set_ylabel("Similarity")
ax2.set_title("B. Alternative Learning Measures")
ax2.legend(loc="upper right")
ax2.set_ylim(0, 1)

# Panel C: Novelty (inverse of similarity)
ax3 = axes[2]
if "novelty_rolling5" in df.columns:
    ax3.plot(df["date"], df["novelty_rolling5"], 'k-', linewidth=1.5, alpha=0.8, label="Novelty (1 - rolling sim)")
add_crisis_shading(ax3, crisis_periods)
ax3.set_ylabel("Novelty")
ax3.set_title("C. Communication Novelty")
ax3.legend(loc="upper right")
ax3.set_xlabel("Date")
ax3.xaxis.set_major_locator(mdates.YearLocator(2))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.suptitle("Figure 1: ECB Communication Similarity Measures Over Time", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig1_similarity_measures.pdf")
plt.savefig(FIGURES_DIR / "fig1_similarity_measures.png")
plt.close()
print("  Saved: fig1_similarity_measures.pdf/png")

# =============================================================================
# FIGURE 2: MARKET RESPONSE EVOLUTION
# =============================================================================
print("[2/8] Generating Figure 2: Market Response Evolution...")

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Panel A: Absolute CAR over time
ax1 = axes[0]
ax1.scatter(df["date"], df["abs_CAR_m5_p5"], alpha=0.6, s=30, c='steelblue', edgecolors='none')
# Rolling average
window = 10
rolling_car = df.set_index("date")["abs_CAR_m5_p5"].rolling(f'{window*30}D', min_periods=3).mean()
ax1.plot(rolling_car.index, rolling_car.values, 'r-', linewidth=2, label=f'Rolling avg ({window} events)')
add_crisis_shading(ax1, crisis_periods)
ax1.set_ylabel("|CAR(-5,+5)|")
ax1.set_title("A. Market Response Magnitude Over Time")
ax1.legend(loc="upper right")

# Panel B: Signed CAR
ax2 = axes[1]
colors = ['green' if x > 0 else 'red' for x in df["CAR_m5_p5"]]
ax2.bar(df["date"], df["CAR_m5_p5"], color=colors, alpha=0.7, width=20)
add_crisis_shading(ax2, crisis_periods)
ax2.axhline(y=0, color='black', linewidth=0.5)
ax2.set_ylabel("CAR(-5,+5)")
ax2.set_title("B. Signed Market Response")
ax2.set_xlabel("Date")
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.suptitle("Figure 2: Market Response to ECB Communication Over Time", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig2_market_response_evolution.pdf")
plt.savefig(FIGURES_DIR / "fig2_market_response_evolution.png")
plt.close()
print("  Saved: fig2_market_response_evolution.pdf/png")

# =============================================================================
# FIGURE 3: CRISIS VS NON-CRISIS COMPARISON
# =============================================================================
print("[3/8] Generating Figure 3: Crisis Regime Comparison...")

if "crisis" in df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    crisis_df = df[df["crisis"] == 1]
    non_crisis_df = df[df["crisis"] == 0]
    
    # Panel A: CAR distribution
    ax1 = axes[0]
    ax1.hist(non_crisis_df["abs_CAR_m5_p5"], bins=20, alpha=0.6, label=f'Non-Crisis (n={len(non_crisis_df)})', color='steelblue')
    ax1.hist(crisis_df["abs_CAR_m5_p5"], bins=20, alpha=0.6, label=f'Crisis (n={len(crisis_df)})', color='crimson')
    ax1.set_xlabel("|CAR(-5,+5)|")
    ax1.set_ylabel("Frequency")
    ax1.set_title("A. Market Response Distribution")
    ax1.legend()
    
    # Panel B: Similarity comparison
    ax2 = axes[1]
    bp_data = [non_crisis_df["similarity_jaccard"].dropna(), crisis_df["similarity_jaccard"].dropna()]
    bp = ax2.boxplot(bp_data, labels=['Non-Crisis', 'Crisis'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('crimson')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax2.set_ylabel("Jaccard Similarity")
    ax2.set_title("B. Communication Similarity")
    
    # Panel C: Pessimism comparison
    ax3 = axes[2]
    bp_data = [non_crisis_df["pessimism_lm"].dropna(), crisis_df["pessimism_lm"].dropna()]
    bp = ax3.boxplot(bp_data, labels=['Non-Crisis', 'Crisis'], patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][1].set_facecolor('crimson')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax3.set_ylabel("Pessimism (LM)")
    ax3.set_title("C. Sentiment")
    
    plt.suptitle("Figure 3: Crisis vs Non-Crisis Regime Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig3_crisis_comparison.pdf")
    plt.savefig(FIGURES_DIR / "fig3_crisis_comparison.png")
    plt.close()
    print("  Saved: fig3_crisis_comparison.pdf/png")

# =============================================================================
# FIGURE 4: RISK AND UNCERTAINTY INDICES OVER TIME
# =============================================================================
print("[4/8] Generating Figure 4: Risk and Uncertainty Indices...")

if "risk_index" in df.columns and "uncertainty_index" in df.columns:
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # Panel A: Risk index
    ax1 = axes[0]
    ax1.plot(df["date"], df["risk_index"] * 100, 'b-', linewidth=1.5, alpha=0.8)
    ax1.fill_between(df["date"], 0, df["risk_index"] * 100, alpha=0.3, color='blue')
    add_crisis_shading(ax1, crisis_periods)
    ax1.set_ylabel("Risk Index (%)")
    ax1.set_title("A. Risk-Related Language Intensity")
    
    # Panel B: Uncertainty index
    ax2 = axes[1]
    ax2.plot(df["date"], df["uncertainty_index"] * 100, 'r-', linewidth=1.5, alpha=0.8)
    ax2.fill_between(df["date"], 0, df["uncertainty_index"] * 100, alpha=0.3, color='red')
    add_crisis_shading(ax2, crisis_periods)
    ax2.set_ylabel("Uncertainty Index (%)")
    ax2.set_title("B. Uncertainty-Related Language Intensity")
    
    # Panel C: Uncertainty-to-Risk Ratio
    ax3 = axes[2]
    if "uncertainty_risk_ratio" in df.columns:
        ratio = df["uncertainty_risk_ratio"].clip(upper=df["uncertainty_risk_ratio"].quantile(0.99))
        ax3.plot(df["date"], ratio, 'purple', linewidth=1.5, alpha=0.8)
        ax3.axhline(y=1, color='gray', linestyle='--', linewidth=1, label='Ratio = 1')
    add_crisis_shading(ax3, crisis_periods)
    ax3.set_ylabel("Uncertainty/Risk Ratio")
    ax3.set_title("C. Relative Uncertainty vs Risk")
    ax3.set_xlabel("Date")
    ax3.legend(loc="upper right")
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.suptitle("Figure 4: Risk vs Uncertainty Language in ECB Communication", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig4_risk_uncertainty_indices.pdf")
    plt.savefig(FIGURES_DIR / "fig4_risk_uncertainty_indices.png")
    plt.close()
    print("  Saved: fig4_risk_uncertainty_indices.pdf/png")

# =============================================================================
# FIGURE 5: RISK VS UNCERTAINTY SCATTER
# =============================================================================
print("[5/8] Generating Figure 5: Risk vs Uncertainty Scatter...")

if "risk_index" in df.columns and "uncertainty_index" in df.columns:
    fig, ax = plt.subplots(figsize=(10, 8))
    
    scatter = ax.scatter(df["risk_index"] * 100, df["uncertainty_index"] * 100, 
                        c=df["abs_CAR_m5_p5"], cmap='RdYlBu_r', 
                        s=50, alpha=0.7, edgecolors='gray', linewidth=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("|CAR(-5,+5)|")
    
    # Add 45-degree line
    max_val = max(df["risk_index"].max(), df["uncertainty_index"].max()) * 100
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='Equal proportion')
    
    ax.set_xlabel("Risk Index (%)")
    ax.set_ylabel("Uncertainty Index (%)")
    ax.set_title("Figure 5: Risk vs Uncertainty Language\n(Color = Market Response Magnitude)")
    ax.legend(loc="upper left")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig5_risk_uncertainty_scatter.pdf")
    plt.savefig(FIGURES_DIR / "fig5_risk_uncertainty_scatter.png")
    plt.close()
    print("  Saved: fig5_risk_uncertainty_scatter.pdf/png")

# =============================================================================
# FIGURE 6: DECOMPOSED SENTIMENT COMPONENTS
# =============================================================================
print("[6/8] Generating Figure 6: Decomposed Sentiment...")

if "pessimism_risk" in df.columns and "pessimism_uncertainty" in df.columns:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Panel A: Stacked area chart of components
    ax1 = axes[0]
    ax1.fill_between(df["date"], 0, df["pessimism_risk"], alpha=0.7, label='Risk component', color='steelblue')
    ax1.fill_between(df["date"], df["pessimism_risk"], 
                    df["pessimism_risk"] + df["pessimism_uncertainty"], 
                    alpha=0.7, label='Uncertainty component', color='crimson')
    if "pessimism_other" in df.columns:
        ax1.fill_between(df["date"], 
                        df["pessimism_risk"] + df["pessimism_uncertainty"],
                        df["pessimism_risk"] + df["pessimism_uncertainty"] + df["pessimism_other"],
                        alpha=0.7, label='Other component', color='gray')
    add_crisis_shading(ax1, crisis_periods, alpha=0.1)
    ax1.set_ylabel("Pessimism Contribution")
    ax1.set_title("A. Decomposed Sentiment Components")
    ax1.legend(loc="upper right")
    
    # Panel B: Comparison with overall pessimism
    ax2 = axes[1]
    ax2.plot(df["date"], df["pessimism_lm"], 'k-', linewidth=2, label='Overall Pessimism (LM)', alpha=0.8)
    ax2.plot(df["date"], df["pessimism_risk"], 'b--', linewidth=1.5, label='Risk Component', alpha=0.7)
    ax2.plot(df["date"], df["pessimism_uncertainty"], 'r--', linewidth=1.5, label='Uncertainty Component', alpha=0.7)
    add_crisis_shading(ax2, crisis_periods)
    ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax2.set_ylabel("Pessimism Score")
    ax2.set_xlabel("Date")
    ax2.set_title("B. Overall vs Decomposed Pessimism")
    ax2.legend(loc="upper right")
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.suptitle("Figure 6: Sentiment Decomposition into Risk and Uncertainty", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "fig6_decomposed_sentiment.pdf")
    plt.savefig(FIGURES_DIR / "fig6_decomposed_sentiment.png")
    plt.close()
    print("  Saved: fig6_decomposed_sentiment.pdf/png")

# =============================================================================
# FIGURE 7: CORRELATION HEATMAP
# =============================================================================
print("[7/8] Generating Figure 7: Correlation Heatmap...")

# Select key variables for correlation
corr_vars = ["abs_CAR_m5_p5", "similarity_jaccard", "pessimism_lm"]

if "risk_index" in df.columns:
    corr_vars.extend(["risk_index", "uncertainty_index"])
if "similarity_rolling5" in df.columns:
    corr_vars.extend(["similarity_rolling5", "similarity_decay"])

available_vars = [v for v in corr_vars if v in df.columns]
corr_matrix = df[available_vars].corr()

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

# Add colorbar
cbar = ax.figure.colorbar(im, ax=ax)
cbar.set_label("Correlation")

# Set ticks
ax.set_xticks(np.arange(len(available_vars)))
ax.set_yticks(np.arange(len(available_vars)))
ax.set_xticklabels(available_vars, rotation=45, ha='right')
ax.set_yticklabels(available_vars)

# Add correlation values
for i in range(len(available_vars)):
    for j in range(len(available_vars)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=9)

ax.set_title("Figure 7: Correlation Matrix of Key Variables")
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig7_correlation_heatmap.pdf")
plt.savefig(FIGURES_DIR / "fig7_correlation_heatmap.png")
plt.close()
print("  Saved: fig7_correlation_heatmap.pdf/png")

# =============================================================================
# FIGURE 8: SUMMARY STATISTICS VISUALIZATION
# =============================================================================
print("[8/8] Generating Figure 8: Summary Statistics...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Panel A: CAR distribution
ax1 = axes[0, 0]
ax1.hist(df["abs_CAR_m5_p5"], bins=30, color='steelblue', alpha=0.7, edgecolor='white')
ax1.axvline(df["abs_CAR_m5_p5"].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["abs_CAR_m5_p5"].mean():.4f}')
ax1.axvline(df["abs_CAR_m5_p5"].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df["abs_CAR_m5_p5"].median():.4f}')
ax1.set_xlabel("|CAR(-5,+5)|")
ax1.set_ylabel("Frequency")
ax1.set_title("A. Distribution of Market Response")
ax1.legend()

# Panel B: Similarity distribution
ax2 = axes[0, 1]
ax2.hist(df["similarity_jaccard"].dropna(), bins=30, color='green', alpha=0.7, edgecolor='white')
ax2.axvline(df["similarity_jaccard"].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["similarity_jaccard"].mean():.3f}')
ax2.set_xlabel("Jaccard Similarity")
ax2.set_ylabel("Frequency")
ax2.set_title("B. Distribution of Communication Similarity")
ax2.legend()

# Panel C: Pessimism distribution
ax3 = axes[1, 0]
ax3.hist(df["pessimism_lm"].dropna(), bins=30, color='purple', alpha=0.7, edgecolor='white')
ax3.axvline(df["pessimism_lm"].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["pessimism_lm"].mean():.3f}')
ax3.axvline(0, color='gray', linestyle='-', linewidth=1)
ax3.set_xlabel("Pessimism (LM)")
ax3.set_ylabel("Frequency")
ax3.set_title("C. Distribution of Sentiment")
ax3.legend()

# Panel D: Time series count by year
ax4 = axes[1, 1]
yearly_counts = df.groupby(df["date"].dt.year).size()
ax4.bar(yearly_counts.index, yearly_counts.values, color='teal', alpha=0.7, edgecolor='white')
ax4.set_xlabel("Year")
ax4.set_ylabel("Number of Events")
ax4.set_title("D. ECB Press Conferences by Year")

plt.suptitle("Figure 8: Summary Statistics of Key Variables", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / "fig8_summary_statistics.pdf")
plt.savefig(FIGURES_DIR / "fig8_summary_statistics.png")
plt.close()
print("  Saved: fig8_summary_statistics.pdf/png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("FIGURE GENERATION COMPLETE")
print("=" * 80)
print(f"\nAll figures saved to: {FIGURES_DIR.resolve()}")
print("\nGenerated files:")
for f in sorted(FIGURES_DIR.glob("*.png")):
    print(f"  - {f.name}")
print("=" * 80)
