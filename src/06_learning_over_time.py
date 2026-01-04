"""
Extension 1: Learning Over Time

This module models how financial markets progressively learn from ECB communication
over time. It implements several mechanisms to capture market learning dynamics:

1. ROLLING WINDOW SIMILARITY: Measures communication novelty relative to recent
   history (last N statements) rather than just the previous statement.

2. CUMULATIVE COMMUNICATION INDEX: Captures accumulated market familiarity with
   ECB language patterns over time.

3. DECAY-WEIGHTED SIMILARITY: Recent communications weighted more heavily than
   distant ones, reflecting fading memory effects.

4. REGIME-BASED SPLITS: Distinguishes between crisis and non-crisis periods
   where market learning dynamics may differ.

Economic Intuition:
-------------------
Markets do not simply react to the previous ECB statement but develop institutional
memory. As ECB communication patterns become more familiar over time:
- Novelty signals become more refined (learning effect)
- Markets may become desensitized to repeated language (habituation)
- Crisis periods may "reset" learning as uncertainty spikes

The learning hypothesis suggests that the relationship between communication
similarity and market reactions should evolve over time as markets accumulate
experience with ECB rhetoric.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import bigrams

# --- Force working directory to project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)

# --- Paths ---
IN_PATH = Path("data/processed/ecb_text_features.csv")
OUT_PATH = Path("data/processed/ecb_text_features.csv")  # overwrite same file

# --- Load data ---
df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

print("=" * 80)
print("EXTENSION 1: LEARNING OVER TIME")
print("=" * 80)
print(f"Loaded {len(df)} observations")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")

# =============================================================================
# 1. ROLLING WINDOW SIMILARITY (Last N=5 statements)
# =============================================================================
print("\n[1/5] Computing rolling window similarity...")

def jaccard_similarity_bigrams(tokens1, tokens2):
    """Compute Jaccard similarity based on bigrams."""
    if not isinstance(tokens1, list) or not isinstance(tokens2, list):
        return np.nan
    b1 = set(bigrams(tokens1))
    b2 = set(bigrams(tokens2))
    if len(b1 | b2) == 0:
        return np.nan
    return len(b1 & b2) / len(b1 | b2)

WINDOW_SIZE = 5  # Rolling window of 5 previous statements

def rolling_window_similarity(df, window_size=WINDOW_SIZE):
    """
    Compute average similarity to the last N statements.
    This captures how novel the current statement is relative to recent history.
    """
    rolling_sim = [np.nan] * window_size
    
    for i in range(window_size, len(df)):
        current_tokens = eval(df.loc[i, "tokens"])
        similarities = []
        
        for j in range(1, window_size + 1):
            past_tokens = eval(df.loc[i - j, "tokens"])
            sim = jaccard_similarity_bigrams(current_tokens, past_tokens)
            if not np.isnan(sim):
                similarities.append(sim)
        
        if similarities:
            rolling_sim.append(np.mean(similarities))
        else:
            rolling_sim.append(np.nan)
    
    return rolling_sim

df["similarity_rolling5"] = rolling_window_similarity(df)

# Novelty is inverse of similarity (how different from recent past)
df["novelty_rolling5"] = 1 - df["similarity_rolling5"]

print(f"  Rolling window similarity computed (window={WINDOW_SIZE})")

# =============================================================================
# 2. CUMULATIVE COMMUNICATION INDEX
# =============================================================================
print("[2/5] Computing cumulative communication index...")

def cumulative_similarity_index(df):
    """
    Cumulative average similarity over all past statements.
    This measures the overall 'learning' level - how much the market
    has been exposed to similar communication patterns.
    """
    cumul_sim = [np.nan]
    
    for i in range(1, len(df)):
        current_tokens = eval(df.loc[i, "tokens"])
        similarities = []
        
        # Compare to all previous statements
        for j in range(i):
            past_tokens = eval(df.loc[j, "tokens"])
            sim = jaccard_similarity_bigrams(current_tokens, past_tokens)
            if not np.isnan(sim):
                similarities.append(sim)
        
        if similarities:
            cumul_sim.append(np.mean(similarities))
        else:
            cumul_sim.append(np.nan)
    
    return cumul_sim

df["cumulative_similarity"] = cumulative_similarity_index(df)

# Market experience proxy: number of ECB statements observed
df["ecb_experience"] = range(len(df))

print("  Cumulative similarity index computed")

# =============================================================================
# 3. DECAY-WEIGHTED SIMILARITY
# =============================================================================
print("[3/5] Computing decay-weighted similarity...")

DECAY_RATE = 0.5  # Half-life decay parameter (lambda)

def decay_weighted_similarity(df, decay_rate=DECAY_RATE):
    """
    Exponentially decay-weighted similarity to past statements.
    Recent statements have more weight than distant ones.
    
    Weight for statement j periods ago: exp(-decay_rate * j)
    
    This reflects the economic intuition that recent communication
    is more salient for market expectations.
    """
    decay_sim = [np.nan]
    
    for i in range(1, len(df)):
        current_tokens = eval(df.loc[i, "tokens"])
        weighted_sum = 0
        weight_sum = 0
        
        for j in range(1, i + 1):  # j periods ago
            past_idx = i - j
            past_tokens = eval(df.loc[past_idx, "tokens"])
            sim = jaccard_similarity_bigrams(current_tokens, past_tokens)
            
            if not np.isnan(sim):
                weight = np.exp(-decay_rate * j)
                weighted_sum += weight * sim
                weight_sum += weight
        
        if weight_sum > 0:
            decay_sim.append(weighted_sum / weight_sum)
        else:
            decay_sim.append(np.nan)
    
    return decay_sim

df["similarity_decay"] = decay_weighted_similarity(df)

print(f"  Decay-weighted similarity computed (decay_rate={DECAY_RATE})")

# =============================================================================
# 4. REGIME-BASED SPLITS (Crisis vs Non-Crisis)
# =============================================================================
print("[4/5] Creating regime indicators...")

# Define crisis periods based on major financial/economic crises
crisis_periods = [
    ("2008-09-15", "2009-06-30"),  # Global Financial Crisis (Lehman to recovery)
    ("2010-05-01", "2012-12-31"),  # European Sovereign Debt Crisis
    ("2020-03-01", "2021-06-30"),  # COVID-19 Crisis
    ("2022-02-24", "2022-12-31"),  # Ukraine War / Energy Crisis
]

def is_crisis_period(date, crisis_periods):
    """Check if date falls within any crisis period."""
    for start, end in crisis_periods:
        if pd.to_datetime(start) <= date <= pd.to_datetime(end):
            return 1
    return 0

df["crisis"] = df["date"].apply(lambda x: is_crisis_period(x, crisis_periods))

# Pre/Post Draghi "Whatever it takes" (July 26, 2012)
df["post_draghi"] = (df["date"] >= "2012-07-26").astype(int)

# Pre/Post Forward Guidance (July 2013)
df["post_forward_guidance"] = (df["date"] >= "2013-07-01").astype(int)

# Early vs Late period (split at median date)
median_date = df["date"].median()
df["late_period"] = (df["date"] >= median_date).astype(int)

print(f"  Crisis periods identified: {df['crisis'].sum()} observations in crisis")
print(f"  Post-Draghi observations: {df['post_draghi'].sum()}")
print(f"  Post-Forward Guidance observations: {df['post_forward_guidance'].sum()}")

# =============================================================================
# 5. TIME-VARYING INTERACTION TERMS
# =============================================================================
print("[5/5] Creating time-varying interaction terms...")

eps = 1e-8

# Log transformations for learning measures
df["log_sim_rolling5"] = np.log(df["similarity_rolling5"] + eps)
df["log_sim_decay"] = np.log(df["similarity_decay"] + eps)
df["log_cumul_sim"] = np.log(df["cumulative_similarity"] + eps)

# Create log_sim_jaccard if it doesn't exist (needed for interactions)
if "log_sim_jaccard" not in df.columns:
    df["log_sim_jaccard"] = np.log(df["similarity_jaccard"] + eps)

# Time trend (normalized)
df["time_trend"] = (df["date"] - df["date"].min()).dt.days
df["time_trend_norm"] = df["time_trend"] / df["time_trend"].max()

# Learning interaction terms
# These capture whether the effect of similarity changes over time
df["sim_x_time"] = df["log_sim_jaccard"] * df["time_trend_norm"]
df["sim_x_experience"] = df["log_sim_jaccard"] * (df["ecb_experience"] / df["ecb_experience"].max())

# Crisis interactions
df["sim_x_crisis"] = df["log_sim_jaccard"] * df["crisis"]
df["pessimism_x_crisis"] = df["pessimism_lm"] * df["crisis"]

# Period interactions
df["sim_x_late"] = df["log_sim_jaccard"] * df["late_period"]
df["pessimism_x_late"] = df["pessimism_lm"] * df["late_period"]

print("  Time-varying interaction terms created")

# =============================================================================
# SAVE ENHANCED DATASET
# =============================================================================
df.to_csv(OUT_PATH, index=False)

print("\n" + "=" * 80)
print("LEARNING OVER TIME FEATURES SUMMARY")
print("=" * 80)
print("\nNew Features Added:")
print("  - similarity_rolling5: Average similarity to last 5 statements")
print("  - novelty_rolling5: 1 - similarity_rolling5 (novelty measure)")
print("  - cumulative_similarity: Average similarity to all past statements")
print("  - ecb_experience: Cumulative count of ECB statements")
print("  - similarity_decay: Decay-weighted similarity (recent = higher weight)")
print("  - crisis: Binary indicator for crisis periods")
print("  - post_draghi: Post 'Whatever it takes' speech indicator")
print("  - post_forward_guidance: Post-July 2013 forward guidance indicator")
print("  - late_period: Second half of sample indicator")
print("  - Various log transforms and interaction terms")
print(f"\nSaved to: {OUT_PATH.resolve()}")
print("=" * 80)

# Display summary statistics
print("\nLearning Measure Summary Statistics:")
learning_cols = ["similarity_rolling5", "novelty_rolling5", "cumulative_similarity", 
                 "similarity_decay", "crisis", "ecb_experience"]
print(df[learning_cols].describe().round(4).to_string())
