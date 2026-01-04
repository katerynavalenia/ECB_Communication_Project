"""
Extension Regressions: Learning Over Time & Risk vs Uncertainty

This script runs regression analyses for both extensions:

EXTENSION 1 - LEARNING OVER TIME:
- Tests whether market sensitivity to ECB communication evolves over time
- Examines rolling window vs cumulative vs decay-weighted similarity
- Regime-based analysis (crisis vs non-crisis)
- Time-varying coefficients through interactions

EXTENSION 2 - RISK VS UNCERTAINTY:
- Decomposes sentiment into risk and uncertainty components
- Tests differential market responses to each component
- Examines risk-uncertainty ratio as predictor

All regressions use OLS with robust standard errors (HC1).
Output tables are LaTeX-exportable and publication-ready.
"""

import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.iolib.summary2 import summary_col

# --- Force working directory to project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)

# --- Paths ---
DATA_PATH = Path("data/processed/regression_dataset.csv")
OUT_LEARNING = Path("outputs/extension1_learning_table.txt")
OUT_LEARNING_LATEX = Path("outputs/extension1_learning_table.tex")
OUT_RISK_UNC = Path("outputs/extension2_risk_uncertainty_table.txt")
OUT_RISK_UNC_LATEX = Path("outputs/extension2_risk_uncertainty_table.tex")
OUT_COMBINED = Path("outputs/extensions_combined_table.txt")
OUT_COMBINED_LATEX = Path("outputs/extensions_combined_table.tex")

# Ensure output directory exists
OUT_LEARNING.parent.mkdir(parents=True, exist_ok=True)

# --- Load data ---
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year

print("=" * 80)
print("EXTENSION REGRESSIONS")
print("=" * 80)
print(f"Loaded {len(df)} observations")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# --- Helper functions ---
def fit_model(data, y, x_cols, name="Model"):
    """Fit OLS with robust SE (HC1) and handle missing data."""
    data_clean = data.dropna(subset=[y.name if hasattr(y, 'name') else 'y'] + x_cols)
    y_clean = data_clean[y.name] if hasattr(y, 'name') else y.loc[data_clean.index]
    X = sm.add_constant(data_clean[x_cols])
    model = sm.OLS(y_clean, X, missing="drop").fit(cov_type="HC1")
    return model

def export_to_latex(table, filepath, caption="", label=""):
    """Export summary table to LaTeX format."""
    latex_str = table.as_latex()
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("% Auto-generated LaTeX table\n")
        f.write("% " + "=" * 70 + "\n")
        f.write(latex_str)
    print(f"  LaTeX table saved: {filepath}")

# =============================================================================
# EXTENSION 1: LEARNING OVER TIME REGRESSIONS
# =============================================================================
print("\n" + "=" * 80)
print("EXTENSION 1: LEARNING OVER TIME")
print("=" * 80)

# Prepare variables
eps = 1e-8
df["time_trend"] = (df["date"] - df["date"].min()).dt.days
df["time_trend_norm"] = df["time_trend"] / df["time_trend"].max()

# Check for learning columns
learning_cols = ["similarity_rolling5", "similarity_decay", "cumulative_similarity",
                 "crisis", "late_period", "ecb_experience"]
missing_cols = [c for c in learning_cols if c not in df.columns]
if missing_cols:
    print(f"WARNING: Missing learning columns: {missing_cols}")
    print("Run 06_learning_over_time.py first")
else:
    # Dependent variable
    y = df["abs_CAR_m5_p5"]
    y.name = "abs_CAR_m5_p5"
    
    # Log transforms
    df["log_sim_rolling5"] = np.log(df["similarity_rolling5"] + eps)
    df["log_sim_decay"] = np.log(df["similarity_decay"] + eps)
    df["log_cumul_sim"] = np.log(df["cumulative_similarity"] + eps)
    
    # Experience normalized
    df["experience_norm"] = df["ecb_experience"] / df["ecb_experience"].max()
    
    # Interaction terms
    df["sim_x_time"] = df["log_sim_jaccard"] * df["time_trend_norm"]
    df["sim_x_crisis"] = df["log_sim_jaccard"] * df["crisis"]
    df["pess_x_crisis"] = df["pessimism_lm"] * df["crisis"]
    df["sim_x_late"] = df["log_sim_jaccard"] * df["late_period"]
    df["pess_x_late"] = df["pessimism_lm"] * df["late_period"]
    
    print("\nFitting Learning Over Time models...")
    
    # L1: Baseline (for comparison)
    print("  Model L1: Baseline...")
    m_l1 = fit_model(df, y, ["log_sim_jaccard", "pessimism_lm", "int_jaccard"])
    
    # L2: Rolling window similarity
    print("  Model L2: Rolling window similarity...")
    m_l2 = fit_model(df, y, ["log_sim_rolling5", "pessimism_lm"])
    
    # L3: Decay-weighted similarity  
    print("  Model L3: Decay-weighted similarity...")
    m_l3 = fit_model(df, y, ["log_sim_decay", "pessimism_lm"])
    
    # L4: Time interaction (learning effect)
    print("  Model L4: Time interaction...")
    m_l4 = fit_model(df, y, ["log_sim_jaccard", "pessimism_lm", "time_trend_norm", "sim_x_time"])
    
    # L5: Crisis regime interaction
    print("  Model L5: Crisis regime...")
    m_l5 = fit_model(df, y, ["log_sim_jaccard", "pessimism_lm", "crisis", 
                             "sim_x_crisis", "pess_x_crisis"])
    
    # L6: Early vs Late period
    print("  Model L6: Early vs Late period...")
    m_l6 = fit_model(df, y, ["log_sim_jaccard", "pessimism_lm", "late_period",
                             "sim_x_late", "pess_x_late"])
    
    # Create summary table
    table_learning = summary_col(
        [m_l1, m_l2, m_l3, m_l4, m_l5, m_l6],
        stars=True,
        model_names=["L1:Base", "L2:Roll", "L3:Decay", "L4:Time", "L5:Crisis", "L6:Period"],
        info_dict={
            "N": lambda x: f"{int(x.nobs)}",
            "R²": lambda x: f"{x.rsquared:.3f}",
            "Adj. R²": lambda x: f"{x.rsquared_adj:.3f}"
        }
    )
    
    print("\n" + "-" * 80)
    print("LEARNING OVER TIME RESULTS")
    print("-" * 80)
    print(table_learning)
    
    # Save text table
    with open(OUT_LEARNING, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("EXTENSION 1: LEARNING OVER TIME REGRESSIONS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Dependent Variable: |CAR(-5,+5)| (Absolute Cumulative Abnormal Return)\n")
        f.write("Estimation: OLS with Robust Standard Errors (HC1)\n\n")
        f.write("Model Specifications:\n")
        f.write("  L1: Baseline (Jaccard similarity)\n")
        f.write("  L2: Rolling window similarity (last 5 statements)\n")
        f.write("  L3: Decay-weighted similarity (exponential decay)\n")
        f.write("  L4: Time trend interaction (learning effect)\n")
        f.write("  L5: Crisis regime interaction\n")
        f.write("  L6: Early vs Late period interaction\n\n")
        f.write(str(table_learning))
        f.write("\n\n")
        f.write("Notes:\n")
        f.write("- log_sim_rolling5: Log of avg similarity to last 5 statements\n")
        f.write("- log_sim_decay: Log of decay-weighted similarity (λ=0.5)\n")
        f.write("- time_trend_norm: Normalized time trend [0,1]\n")
        f.write("- sim_x_time: Interaction (similarity × time trend)\n")
        f.write("- crisis: Binary indicator for crisis periods\n")
        f.write("- late_period: Second half of sample indicator\n")
        f.write("- *** p<0.01, ** p<0.05, * p<0.1\n")
    
    print(f"\nSaved: {OUT_LEARNING.resolve()}")
    
    # Export LaTeX
    export_to_latex(table_learning, OUT_LEARNING_LATEX,
                   caption="Learning Over Time: Market Response to ECB Communication",
                   label="tab:learning")

# =============================================================================
# EXTENSION 2: RISK VS UNCERTAINTY REGRESSIONS
# =============================================================================
print("\n" + "=" * 80)
print("EXTENSION 2: RISK VS UNCERTAINTY")
print("=" * 80)

# Check for risk/uncertainty columns
ru_cols = ["risk_index", "uncertainty_index", "pessimism_risk", "pessimism_uncertainty"]
missing_ru = [c for c in ru_cols if c not in df.columns]
if missing_ru:
    print(f"WARNING: Missing risk/uncertainty columns: {missing_ru}")
    print("Run 04b_risk_uncertainty.py first")
else:
    y = df["abs_CAR_m5_p5"]
    y.name = "abs_CAR_m5_p5"
    
    # Standardize indices for interpretability
    df["risk_z"] = (df["risk_index"] - df["risk_index"].mean()) / df["risk_index"].std()
    df["unc_z"] = (df["uncertainty_index"] - df["uncertainty_index"].mean()) / df["uncertainty_index"].std()
    
    # Uncertainty-to-risk ratio (log transformed for interpretability)
    df["unc_risk_ratio"] = df["uncertainty_index"] / (df["risk_index"] + eps)
    df["log_unc_risk_ratio"] = np.log(df["unc_risk_ratio"] + eps)
    
    # Interactions with similarity
    df["int_jaccard_risk"] = df["log_sim_jaccard"] * df["risk_z"]
    df["int_jaccard_unc"] = df["log_sim_jaccard"] * df["unc_z"]
    
    print("\nFitting Risk vs Uncertainty models...")
    
    # U1: Baseline (for comparison)
    print("  Model U1: Baseline (overall pessimism)...")
    m_u1 = fit_model(df, y, ["log_sim_jaccard", "pessimism_lm", "int_jaccard"])
    
    # U2: Risk index only
    print("  Model U2: Risk index only...")
    m_u2 = fit_model(df, y, ["log_sim_jaccard", "risk_z"])
    
    # U3: Uncertainty index only
    print("  Model U3: Uncertainty index only...")
    m_u3 = fit_model(df, y, ["log_sim_jaccard", "unc_z"])
    
    # U4: Both indices (horse race)
    print("  Model U4: Risk vs Uncertainty (horse race)...")
    m_u4 = fit_model(df, y, ["log_sim_jaccard", "risk_z", "unc_z"])
    
    # U5: Decomposed pessimism components
    print("  Model U5: Decomposed pessimism...")
    m_u5 = fit_model(df, y, ["log_sim_jaccard", "pessimism_risk", "pessimism_uncertainty", "pessimism_other"])
    
    # U6: With similarity interactions
    print("  Model U6: With similarity interactions...")
    m_u6 = fit_model(df, y, ["log_sim_jaccard", "risk_z", "unc_z", 
                             "int_jaccard_risk", "int_jaccard_unc"])
    
    # U7: Uncertainty-risk ratio
    print("  Model U7: Uncertainty-risk ratio...")
    m_u7 = fit_model(df, y, ["log_sim_jaccard", "pessimism_lm", "log_unc_risk_ratio"])
    
    # Create summary table
    table_ru = summary_col(
        [m_u1, m_u2, m_u3, m_u4, m_u5, m_u6, m_u7],
        stars=True,
        model_names=["U1:Base", "U2:Risk", "U3:Unc", "U4:Both", "U5:Decomp", "U6:Int", "U7:Ratio"],
        info_dict={
            "N": lambda x: f"{int(x.nobs)}",
            "R²": lambda x: f"{x.rsquared:.3f}",
            "Adj. R²": lambda x: f"{x.rsquared_adj:.3f}"
        }
    )
    
    print("\n" + "-" * 80)
    print("RISK VS UNCERTAINTY RESULTS")
    print("-" * 80)
    print(table_ru)
    
    # Save text table
    with open(OUT_RISK_UNC, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("EXTENSION 2: RISK VS UNCERTAINTY REGRESSIONS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Dependent Variable: |CAR(-5,+5)| (Absolute Cumulative Abnormal Return)\n")
        f.write("Estimation: OLS with Robust Standard Errors (HC1)\n\n")
        f.write("Model Specifications:\n")
        f.write("  U1: Baseline (overall pessimism)\n")
        f.write("  U2: Risk index only\n")
        f.write("  U3: Uncertainty index only\n")
        f.write("  U4: Risk vs Uncertainty (horse race)\n")
        f.write("  U5: Decomposed pessimism (risk + uncertainty + other)\n")
        f.write("  U6: With similarity × risk/uncertainty interactions\n")
        f.write("  U7: Uncertainty-to-risk ratio\n\n")
        f.write(str(table_ru))
        f.write("\n\n")
        f.write("Notes:\n")
        f.write("- risk_z: Standardized risk language index\n")
        f.write("- unc_z: Standardized uncertainty language index\n")
        f.write("- pessimism_risk: Pessimism from risk-related words\n")
        f.write("- pessimism_uncertainty: Pessimism from uncertainty-related words\n")
        f.write("- pessimism_other: Residual pessimism\n")
        f.write("- log_unc_risk_ratio: Log of uncertainty/risk language ratio\n")
        f.write("- *** p<0.01, ** p<0.05, * p<0.1\n")
    
    print(f"\nSaved: {OUT_RISK_UNC.resolve()}")
    
    # Export LaTeX
    export_to_latex(table_ru, OUT_RISK_UNC_LATEX,
                   caption="Risk vs Uncertainty: Decomposing ECB Sentiment",
                   label="tab:risk_uncertainty")

# =============================================================================
# COMBINED SPECIFICATION (BOTH EXTENSIONS)
# =============================================================================
print("\n" + "=" * 80)
print("COMBINED SPECIFICATION")
print("=" * 80)

if not missing_cols and not missing_ru:
    y = df["abs_CAR_m5_p5"]
    y.name = "abs_CAR_m5_p5"
    
    print("\nFitting combined models...")
    
    # C1: Baseline
    print("  Model C1: Baseline...")
    m_c1 = fit_model(df, y, ["log_sim_jaccard", "pessimism_lm", "int_jaccard"])
    
    # C2: Learning + Risk/Unc decomposition
    print("  Model C2: Learning + Decomposition...")
    m_c2 = fit_model(df, y, ["log_sim_decay", "risk_z", "unc_z", "time_trend_norm"])
    
    # C3: Full model with crisis
    print("  Model C3: Full model with crisis...")
    m_c3 = fit_model(df, y, ["log_sim_decay", "risk_z", "unc_z", "crisis",
                             "time_trend_norm", "n_tokens"])
    
    # C4: Kitchen sink (all interactions)
    print("  Model C4: Kitchen sink...")
    m_c4 = fit_model(df, y, ["log_sim_jaccard", "risk_z", "unc_z", "crisis",
                             "time_trend_norm", "sim_x_crisis", "sim_x_time"])
    
    # Create summary table
    table_combined = summary_col(
        [m_c1, m_c2, m_c3, m_c4],
        stars=True,
        model_names=["C1:Base", "C2:Learn+RU", "C3:Full", "C4:Kitchen"],
        info_dict={
            "N": lambda x: f"{int(x.nobs)}",
            "R²": lambda x: f"{x.rsquared:.3f}",
            "Adj. R²": lambda x: f"{x.rsquared_adj:.3f}"
        }
    )
    
    print("\n" + "-" * 80)
    print("COMBINED SPECIFICATION RESULTS")
    print("-" * 80)
    print(table_combined)
    
    # Save combined table
    with open(OUT_COMBINED, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("COMBINED EXTENSIONS: LEARNING + RISK/UNCERTAINTY\n")
        f.write("=" * 80 + "\n\n")
        f.write("Dependent Variable: |CAR(-5,+5)| (Absolute Cumulative Abnormal Return)\n")
        f.write("Estimation: OLS with Robust Standard Errors (HC1)\n\n")
        f.write(str(table_combined))
        f.write("\n\n")
        f.write("Notes:\n")
        f.write("- C1: Baseline replication specification\n")
        f.write("- C2: Decay-weighted similarity + risk/uncertainty decomposition\n")
        f.write("- C3: Full model with crisis regime and controls\n")
        f.write("- C4: Kitchen sink with all interactions\n")
        f.write("- *** p<0.01, ** p<0.05, * p<0.1\n")
    
    print(f"\nSaved: {OUT_COMBINED.resolve()}")
    
    # Export LaTeX
    export_to_latex(table_combined, OUT_COMBINED_LATEX,
                   caption="Combined Extensions: Learning and Risk/Uncertainty Decomposition",
                   label="tab:combined")

# =============================================================================
# SAVE FULL REGRESSION DATASET
# =============================================================================
OUT_DATASET = Path("data/processed/regression_dataset_extended.csv")
df.to_csv(OUT_DATASET, index=False)
print(f"\nFull extended dataset saved: {OUT_DATASET.resolve()}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 80)
print("EXTENSION REGRESSIONS COMPLETE")
print("=" * 80)
print("\nOUTPUT FILES:")
print(f"  Extension 1 (Learning): {OUT_LEARNING.resolve()}")
print(f"  Extension 1 (LaTeX):    {OUT_LEARNING_LATEX.resolve()}")
print(f"  Extension 2 (Risk/Unc): {OUT_RISK_UNC.resolve()}")
print(f"  Extension 2 (LaTeX):    {OUT_RISK_UNC_LATEX.resolve()}")
print(f"  Combined table:         {OUT_COMBINED.resolve()}")
print(f"  Combined (LaTeX):       {OUT_COMBINED_LATEX.resolve()}")
print(f"  Extended dataset:       {OUT_DATASET.resolve()}")
print("=" * 80)
