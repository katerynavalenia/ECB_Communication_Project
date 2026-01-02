import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

# --- Force working directory to project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)

# --- Paths ---
DATA_PATH = Path('data/processed/event_study_constant_mean.csv')
OUT_DATASET = Path("data/processed/regression_dataset.csv")
OUT_REPLICATION = Path("outputs/replication_table.txt")
OUT_EXTENSION = Path("outputs/extension_table.txt")
OUT_REPLICATION.parent.mkdir(parents=True, exist_ok=True)
OUT_EXTENSION.parent.mkdir(parents=True, exist_ok=True)

# --- Load data ---
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year

# --- Prepare regression variables ---
# time trend
df["time"] = (df["date"] - df["date"].min()).dt.days

# log similarity (avoid log(0))
eps = 1e-8
df["log_sim_jaccard"] = np.log(df["similarity_jaccard"] + eps)
df["log_sim_cosine"] = np.log(df["similarity_cosine"] + eps)

# interactions
df["int_jaccard"] = df["log_sim_jaccard"] * df["pessimism_lm"]
df["int_cosine"] = df["log_sim_cosine"] * df["pessimism_lm"]

# Save full regression dataset
df.to_csv(OUT_DATASET, index=False)
print("="*80)
print("REGRESSION DATASET PREPARED")
print("="*80)
print(f"Total observations: {len(df)}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Year range: {df['year'].min()} to {df['year'].max()}")
print(f"Saved: {OUT_DATASET.resolve()}")
print("="*80)

# =============================================================================
# REPLICATION BENCHMARK (overlap period: 2007-2013)
# =============================================================================
print("\n" + "="*80)
print("REPLICATION BENCHMARK: 2007-2013 (overlap period)")
print("="*80)

# Filter to replication period (overlap between paper and Yahoo Finance data)
df_repl = df[df["date"] <= "2013-12-31"].copy()
print(f"Replication sample: {len(df_repl)} observations")
print(f"Date range: {df_repl['date'].min()} to {df_repl['date'].max()}")
print("="*80)

# dependent variable
y_repl = df_repl["abs_CAR_m5_p5"]

def fit_model(data, y, x_cols):
    """Fit OLS with robust SE (HC1)"""
    X = sm.add_constant(data[x_cols])
    return sm.OLS(y, X, missing="drop").fit(cov_type="HC1")

# Model R1: Baseline replication (Jaccard similarity as in paper)
print("\nFitting Model R1: Jaccard baseline...")
m_r1 = fit_model(df_repl, y_repl, ["log_sim_jaccard", "pessimism_lm", "int_jaccard"])

# Model R2: Robustness with Cosine similarity
print("Fitting Model R2: Cosine baseline (robustness)...")
m_r2 = fit_model(df_repl, y_repl, ["log_sim_cosine", "pessimism_lm", "int_cosine"])

# Print individual summaries
print("\n" + "-"*80)
print("Model R1: Jaccard Baseline")
print("-"*80)
print(m_r1.summary())

print("\n" + "-"*80)
print("Model R2: Cosine Baseline (Robustness)")
print("-"*80)
print(m_r2.summary())

# Create replication table
table_repl = summary_col(
    [m_r1, m_r2],
    stars=True,
    model_names=["R1: Jaccard", "R2: Cosine"],
    info_dict={
        "N": lambda x: f"{int(x.nobs)}",
        "R²": lambda x: f"{x.rsquared:.3f}",
        "Adj. R²": lambda x: f"{x.rsquared_adj:.3f}"
    }
)

print("\n" + "="*80)
print("REPLICATION TABLE (2007-2013)")
print("="*80)
print(table_repl)
print("="*80)

# Save replication table
with open(OUT_REPLICATION, "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("REPLICATION BENCHMARK: 2007-2013 (Overlap Period)\n")
    f.write("="*80 + "\n\n")
    f.write("Sample: Events with date <= 2013-12-31\n")
    f.write(f"N = {len(df_repl)} events\n")
    f.write(f"Date range: {df_repl['date'].min()} to {df_repl['date'].max()}\n\n")
    f.write("Dependent Variable: abs_CAR_m5_p5 (Absolute Cumulative Abnormal Return)\n")
    f.write("Estimation: OLS with Robust Standard Errors (HC1)\n\n")
    f.write(str(table_repl))
    f.write("\n\n")
    f.write("Notes:\n")
    f.write("- R1 uses Jaccard bigram similarity (as in original paper)\n")
    f.write("- R2 uses Cosine similarity (robustness check)\n")
    f.write("- log_sim_*: Log of similarity measure\n")
    f.write("- pessimism_lm: Loughran-McDonald pessimism = (neg-pos)/(neg+pos)\n")
    f.write("- int_*: Interaction term (log_similarity × pessimism_lm)\n")
    f.write("- *** p<0.01, ** p<0.05, * p<0.1\n")

print(f"\nSaved: {OUT_REPLICATION.resolve()}")

# =============================================================================
# EXTENSION (2007-2025 full sample)
# =============================================================================
print("\n" + "="*80)
print("EXTENSION: 2007-2025 (Full Sample)")
print("="*80)

# Use full sample (all available data)
df_ext = df.copy()
print(f"Extension sample: {len(df_ext)} observations")
print(f"Date range: {df_ext['date'].min()} to {df_ext['date'].max()}")
print("="*80)

# dependent variable
y_ext = df_ext["abs_CAR_m5_p5"]

# Model E1: Jaccard baseline
print("\nFitting Model E1: Jaccard baseline...")
m_e1 = fit_model(df_ext, y_ext, ["log_sim_jaccard", "pessimism_lm", "int_jaccard"])

# Model E2: Jaccard + controls
print("Fitting Model E2: Jaccard + controls...")
m_e2 = fit_model(df_ext, y_ext, ["log_sim_jaccard", "pessimism_lm", "int_jaccard", "n_tokens", "time"])

# Model E3: Cosine baseline
print("Fitting Model E3: Cosine baseline...")
m_e3 = fit_model(df_ext, y_ext, ["log_sim_cosine", "pessimism_lm", "int_cosine"])

# Model E4: Cosine + controls
print("Fitting Model E4: Cosine + controls...")
m_e4 = fit_model(df_ext, y_ext, ["log_sim_cosine", "pessimism_lm", "int_cosine", "n_tokens", "time"])

# Print individual summaries
print("\n" + "-"*80)
print("Model E1: Jaccard Baseline")
print("-"*80)
print(m_e1.summary())

print("\n" + "-"*80)
print("Model E2: Jaccard + Controls")
print("-"*80)
print(m_e2.summary())

print("\n" + "-"*80)
print("Model E3: Cosine Baseline")
print("-"*80)
print(m_e3.summary())

print("\n" + "-"*80)
print("Model E4: Cosine + Controls")
print("-"*80)
print(m_e4.summary())

# Create extension table
table_ext = summary_col(
    [m_e1, m_e2, m_e3, m_e4],
    stars=True,
    model_names=["E1: Jaccard", "E2: Jac+Ctrl", "E3: Cosine", "E4: Cos+Ctrl"],
    info_dict={
        "N": lambda x: f"{int(x.nobs)}",
        "R²": lambda x: f"{x.rsquared:.3f}",
        "Adj. R²": lambda x: f"{x.rsquared_adj:.3f}"
    }
)

print("\n" + "="*80)
print("EXTENSION TABLE (2007-2025)")
print("="*80)
print(table_ext)
print("="*80)

# Save extension table
with open(OUT_EXTENSION, "w", encoding="utf-8") as f:
    f.write("="*80 + "\n")
    f.write("EXTENSION: 2007-2025 (Full Sample Period)\n")
    f.write("="*80 + "\n\n")
    f.write("Sample: All available events (full sample)\n")
    f.write(f"N = {len(df_ext)} events\n")
    f.write(f"Date range: {df_ext['date'].min()} to {df_ext['date'].max()}\n\n")
    f.write("Dependent Variable: abs_CAR_m5_p5 (Absolute Cumulative Abnormal Return)\n")
    f.write("Estimation: OLS with Robust Standard Errors (HC1)\n\n")
    f.write(str(table_ext))
    f.write("\n\n")
    f.write("Notes:\n")
    f.write("- E1: Jaccard baseline (extended period)\n")
    f.write("- E2: Jaccard + controls (n_tokens, time)\n")
    f.write("- E3: Cosine baseline (alternative similarity)\n")
    f.write("- E4: Cosine + controls (n_tokens, time)\n")
    f.write("- log_sim_*: Log of similarity measure\n")
    f.write("- pessimism_lm: Loughran-McDonald pessimism = (neg-pos)/(neg+pos)\n")
    f.write("- int_*: Interaction term (log_similarity × pessimism_lm)\n")
    f.write("- n_tokens: Document length (number of tokens)\n")
    f.write("- time: Time trend (days since first event)\n")
    f.write("- *** p<0.01, ** p<0.05, * p<0.1\n")

print(f"\nSaved: {OUT_EXTENSION.resolve()}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*80)
print("REGRESSION ANALYSIS COMPLETE")
print("="*80)
print("\nSAMPLE SIZES:")
print(f"  Replication (2007-2013): {len(df_repl)} events")
print(f"  Extension (2007-2025):   {len(df_ext)} events")
print("\nDATE RANGES:")
print(f"  Replication: {df_repl['date'].min()} to {df_repl['date'].max()}")
print(f"  Extension:   {df_ext['date'].min()} to {df_ext['date'].max()}")
print("\nOUTPUT FILES:")
print(f"  Replication table: {OUT_REPLICATION.resolve()}")
print(f"  Extension table:   {OUT_EXTENSION.resolve()}")
print(f"  Full dataset:      {OUT_DATASET.resolve()}")
print("="*80)
