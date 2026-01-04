"""
Extension 2: Separating Risk vs. Uncertainty in Sentiment

This module decomposes the general sentiment measure into two distinct components:
1. RISK INDEX: Language related to quantifiable, probabilistic risks
2. UNCERTAINTY INDEX: Language related to Knightian uncertainty (unmeasurable)

Economic Intuition:
-------------------
Following Knight (1921) and the uncertainty literature:
- RISK refers to situations where probabilities are known/estimable
  (e.g., "probability", "expected", "forecast", "estimated")
- UNCERTAINTY refers to situations where probabilities cannot be estimated
  (e.g., "uncertain", "unpredictable", "unknown", "unclear")

This distinction matters for financial markets because:
- Risk can be priced and hedged through standard financial instruments
- Uncertainty leads to volatility premium and risk-aversion spikes
- Baker, Bloom & Davis (2016) show uncertainty shocks have real effects

The decomposition allows testing whether markets respond differently to:
1. Risk-related pessimism (quantifiable downside)
2. Uncertainty-related pessimism (unquantifiable ambiguity)

Methodology:
------------
1. Create domain-specific dictionaries for risk vs uncertainty terms
2. Use Loughran-McDonald negative words as base for sentiment
3. Classify negative words by semantic category (risk vs uncertainty)
4. Construct separate indices and test differential market effects
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize

# --- Force working directory to project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)

# --- Paths ---
IN_PATH = Path("data/processed/ecb_text_features.csv")
LM_PATH = Path("data/raw/Loughran-McDonald_MasterDictionary_1993-2024.csv")
OUT_PATH = Path("data/processed/ecb_text_features.csv")  # overwrite same file

# --- Load data ---
df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])

print("=" * 80)
print("EXTENSION 2: SEPARATING RISK VS UNCERTAINTY IN SENTIMENT")
print("=" * 80)
print(f"Loaded {len(df)} observations")

# --- Load Loughran-McDonald Dictionary ---
lm = pd.read_csv(LM_PATH)
lm_uncertainty_words = lm.loc[lm["Uncertainty"] > 0, "Word"].str.lower().tolist()

# =============================================================================
# DICTIONARY DEFINITIONS
# =============================================================================

# RISK DICTIONARY
# Terms associated with quantifiable, probabilistic assessments
RISK_TERMS = {
    # Core risk terminology
    "risk", "risks", "risky", "riskier", "riskiness",
    "probability", "probabilities", "probable", "probably",
    "likelihood", "likely", "unlikely",
    "expected", "expectation", "expectations", "expect", "expects",
    "forecast", "forecasts", "forecasted", "forecasting",
    "predict", "predicts", "predicted", "prediction", "predictions",
    "estimate", "estimates", "estimated", "estimation",
    "scenario", "scenarios", "baseline", "projection", "projections",
    "model", "models", "modeled", "modelling",
    "variance", "volatility", "deviation", "deviations",
    "exposure", "exposures", "exposed",
    "downside", "upside", "tail",
    "hedge", "hedging", "hedged",
    "stress", "stressed", "stresses",
    "default", "defaults", "defaulted",
    "credit", "creditworthy", "creditworthiness",
    "leverage", "leveraged", "leveraging",
    "contagion", "spillover", "spillovers",
    "systemic", "systematic",
    # Financial risk terms
    "loss", "losses", "impairment", "impairments",
    "writedown", "writedowns", "writeoff",
    "provision", "provisions", "provisioning",
    "buffer", "buffers", "capital",
    "solvency", "solvent", "insolvent",
    "liquidity", "liquid", "illiquid", "illiquidity",
}

# UNCERTAINTY DICTIONARY
# Terms associated with Knightian uncertainty (unmeasurable/unknown)
UNCERTAINTY_TERMS = {
    # Core uncertainty terminology
    "uncertainty", "uncertainties", "uncertain",
    "unpredictable", "unpredictability",
    "unknown", "unknowns", "unknowable",
    "unclear", "unclarity",
    "ambiguous", "ambiguity", "ambiguities",
    "vague", "vagueness",
    "imprecise", "imprecision",
    "indefinite", "indefinitely",
    "indeterminate",
    "unforeseeable", "unforeseen",
    "unexpected", "unexpectedly",
    "surprise", "surprised", "surprising", "surprisingly",
    "shock", "shocks", "shocked",
    "unprecedented", "extraordinary",
    "exceptional", "exceptionally",
    "unusual", "unusually",
    "abnormal", "abnormally",
    "volatile", "volatility",
    "turbulence", "turbulent",
    "disruption", "disruptions", "disruptive",
    "instability", "unstable",
    "unrest",
    "turmoil",
    "chaos", "chaotic",
    "confusion", "confused", "confusing",
    "complexity", "complex", "complexities",
    "opaque", "opacity",
    "doubt", "doubts", "doubtful",
    "hesitant", "hesitancy", "hesitate",
    "cautious", "caution", "cautiously",
    "wary", "wariness",
    "skeptical", "skepticism",
    "tentative", "tentatively",
    # LM Uncertainty words (add from dictionary)
    *[w.lower() for w in lm_uncertainty_words],
}

# Remove overlaps (prioritize uncertainty classification)
RISK_TERMS_CLEAN = RISK_TERMS - UNCERTAINTY_TERMS

print(f"\nDictionary sizes:")
print(f"  Risk terms: {len(RISK_TERMS_CLEAN)}")
print(f"  Uncertainty terms: {len(UNCERTAINTY_TERMS)}")

# =============================================================================
# SENTIMENT DECOMPOSITION FUNCTIONS
# =============================================================================

# Load full LM dictionary for positive/negative classification
neg_words = set(lm.loc[lm["Negative"] > 0, "Word"].str.lower().tolist())
pos_words = set(lm.loc[lm["Positive"] > 0, "Word"].str.lower().tolist())

def compute_risk_index(text):
    """
    Compute Risk Index: proportion of risk-related terms in document.
    Higher values = more risk-related language.
    """
    if not isinstance(text, str):
        return 0.0
    tokens = word_tokenize(text.lower())
    if len(tokens) == 0:
        return 0.0
    risk_count = sum(1 for t in tokens if t in RISK_TERMS_CLEAN)
    return risk_count / len(tokens)

def compute_uncertainty_index(text):
    """
    Compute Uncertainty Index: proportion of uncertainty-related terms.
    Higher values = more uncertainty-related language.
    """
    if not isinstance(text, str):
        return 0.0
    tokens = word_tokenize(text.lower())
    if len(tokens) == 0:
        return 0.0
    unc_count = sum(1 for t in tokens if t in UNCERTAINTY_TERMS)
    return unc_count / len(tokens)

def compute_risk_sentiment(text):
    """
    Compute Risk-Weighted Sentiment:
    Pessimism score weighted by risk-related language intensity.
    
    = (neg_risk - pos_risk) / (neg_risk + pos_risk)
    where neg_risk = negative words that are also risk-related
    """
    if not isinstance(text, str):
        return 0.0
    tokens = word_tokenize(text.lower())
    
    # Find words that are both negative AND risk-related
    neg_risk = sum(1 for t in tokens if t in neg_words and t in RISK_TERMS_CLEAN)
    pos_risk = sum(1 for t in tokens if t in pos_words and t in RISK_TERMS_CLEAN)
    
    total = neg_risk + pos_risk
    if total == 0:
        return 0.0
    return (neg_risk - pos_risk) / total

def compute_uncertainty_sentiment(text):
    """
    Compute Uncertainty-Weighted Sentiment:
    Pessimism score weighted by uncertainty-related language intensity.
    
    = (neg_unc - pos_unc) / (neg_unc + pos_unc)
    where neg_unc = negative words that are also uncertainty-related
    """
    if not isinstance(text, str):
        return 0.0
    tokens = word_tokenize(text.lower())
    
    # Find words that are both negative AND uncertainty-related
    neg_unc = sum(1 for t in tokens if t in neg_words and t in UNCERTAINTY_TERMS)
    pos_unc = sum(1 for t in tokens if t in pos_words and t in UNCERTAINTY_TERMS)
    
    total = neg_unc + pos_unc
    if total == 0:
        return 0.0
    return (neg_unc - pos_unc) / total

def compute_sentiment_components(text):
    """
    Decompose overall pessimism into risk and uncertainty components.
    
    Returns:
        pessimism_risk: Contribution to pessimism from risk words
        pessimism_uncertainty: Contribution to pessimism from uncertainty words
        pessimism_other: Residual pessimism (neither risk nor uncertainty)
    """
    if not isinstance(text, str):
        return 0.0, 0.0, 0.0
    
    tokens = word_tokenize(text.lower())
    
    # Categorize negative words
    neg_risk = sum(1 for t in tokens if t in neg_words and t in RISK_TERMS_CLEAN)
    neg_unc = sum(1 for t in tokens if t in neg_words and t in UNCERTAINTY_TERMS)
    neg_other = sum(1 for t in tokens if t in neg_words and t not in RISK_TERMS_CLEAN and t not in UNCERTAINTY_TERMS)
    
    # Categorize positive words
    pos_risk = sum(1 for t in tokens if t in pos_words and t in RISK_TERMS_CLEAN)
    pos_unc = sum(1 for t in tokens if t in pos_words and t in UNCERTAINTY_TERMS)
    pos_other = sum(1 for t in tokens if t in pos_words and t not in RISK_TERMS_CLEAN and t not in UNCERTAINTY_TERMS)
    
    total_neg = neg_risk + neg_unc + neg_other
    total_pos = pos_risk + pos_unc + pos_other
    total = total_neg + total_pos
    
    if total == 0:
        return 0.0, 0.0, 0.0
    
    # Proportional contributions to pessimism
    pessimism_risk = (neg_risk - pos_risk) / total
    pessimism_unc = (neg_unc - pos_unc) / total
    pessimism_other = (neg_other - pos_other) / total
    
    return pessimism_risk, pessimism_unc, pessimism_other

# =============================================================================
# APPLY TO DATA
# =============================================================================
print("\nComputing indices...")

# Raw indices (proportion of document)
print("  [1/4] Computing Risk Index...")
df["risk_index"] = df["content_raw"].apply(compute_risk_index)

print("  [2/4] Computing Uncertainty Index...")
df["uncertainty_index"] = df["content_raw"].apply(compute_uncertainty_index)

# Sentiment decomposition
print("  [3/4] Decomposing sentiment components...")
components = df["content_raw"].apply(compute_sentiment_components)
df["pessimism_risk"] = components.apply(lambda x: x[0])
df["pessimism_uncertainty"] = components.apply(lambda x: x[1])
df["pessimism_other"] = components.apply(lambda x: x[2])

# Cross-weighted sentiments
print("  [4/4] Computing cross-weighted sentiments...")
df["risk_sentiment"] = df["content_raw"].apply(compute_risk_sentiment)
df["uncertainty_sentiment"] = df["content_raw"].apply(compute_uncertainty_sentiment)

# =============================================================================
# DERIVED MEASURES
# =============================================================================

# Uncertainty-to-Risk Ratio (higher = more uncertainty relative to risk)
df["uncertainty_risk_ratio"] = df["uncertainty_index"] / (df["risk_index"] + 1e-8)

# Standardized indices (z-scores for comparability)
df["risk_index_z"] = (df["risk_index"] - df["risk_index"].mean()) / df["risk_index"].std()
df["uncertainty_index_z"] = (df["uncertainty_index"] - df["uncertainty_index"].mean()) / df["uncertainty_index"].std()

# Interaction terms with similarity
eps = 1e-8
if "log_sim_jaccard" in df.columns:
    df["int_jaccard_risk"] = df["log_sim_jaccard"] * df["risk_index_z"]
    df["int_jaccard_uncertainty"] = df["log_sim_jaccard"] * df["uncertainty_index_z"]
    df["int_jaccard_unc_ratio"] = df["log_sim_jaccard"] * df["uncertainty_risk_ratio"]

# =============================================================================
# SAVE ENHANCED DATASET
# =============================================================================
df.to_csv(OUT_PATH, index=False)

print("\n" + "=" * 80)
print("RISK VS UNCERTAINTY FEATURES SUMMARY")
print("=" * 80)
print("\nNew Features Added:")
print("  - risk_index: Proportion of risk-related terms (raw)")
print("  - uncertainty_index: Proportion of uncertainty-related terms (raw)")
print("  - pessimism_risk: Pessimism contribution from risk words")
print("  - pessimism_uncertainty: Pessimism contribution from uncertainty words")
print("  - pessimism_other: Residual pessimism (neither risk nor uncertainty)")
print("  - risk_sentiment: Sentiment among risk-related words only")
print("  - uncertainty_sentiment: Sentiment among uncertainty-related words only")
print("  - uncertainty_risk_ratio: Relative uncertainty vs risk language")
print("  - risk_index_z, uncertainty_index_z: Standardized indices")
print("  - Interaction terms with similarity measure")
print(f"\nSaved to: {OUT_PATH.resolve()}")
print("=" * 80)

# Display summary statistics
print("\nRisk vs Uncertainty Summary Statistics:")
ru_cols = ["risk_index", "uncertainty_index", "uncertainty_risk_ratio",
           "pessimism_risk", "pessimism_uncertainty", "pessimism_other"]
print(df[ru_cols].describe().round(6).to_string())

# Verify decomposition: pessimism_lm ≈ pessimism_risk + pessimism_uncertainty + pessimism_other
if "pessimism_lm" in df.columns:
    df["pessimism_check"] = df["pessimism_risk"] + df["pessimism_uncertainty"] + df["pessimism_other"]
    correlation = df["pessimism_lm"].corr(df["pessimism_check"])
    print(f"\nDecomposition validation (pessimism_lm vs sum of components):")
    print(f"  Correlation: {correlation:.4f}")

# Correlation matrix between key measures
print("\nCorrelation Matrix (Risk vs Uncertainty measures):")
corr_cols = ["pessimism_lm", "risk_index", "uncertainty_index", 
             "pessimism_risk", "pessimism_uncertainty"]
print(df[corr_cols].corr().round(3).to_string())
