# ECB Communication Project - Extensions Documentation

This document describes two new extensions to the ECB Communication Project, implementing **Learning Over Time** and **Risk vs. Uncertainty Decomposition**.

---

## Table of Contents

1. [Extension 1: Learning Over Time](#extension-1-learning-over-time)
2. [Extension 2: Risk vs. Uncertainty Decomposition](#extension-2-risk-vs-uncertainty-decomposition)
3. [Implementation Files](#implementation-files)
4. [Output Files](#output-files)
5. [Running the Extensions](#running-the-extensions)
6. [Regression Specifications](#regression-specifications)

---

## Extension 1: Learning Over Time

### Economic Intuition

Financial markets do not simply react to the most recent ECB statement in isolation. Instead, they develop **institutional memory** through repeated exposure to ECB communication patterns. This extension models how market sensitivity to ECB communication evolves over time.

Key hypotheses:
- **Learning Effect**: As markets gain experience with ECB rhetoric, they become more efficient at processing information, potentially reducing market response to predictable content.
- **Habituation**: Repeated similar language may lead to desensitization, reducing market reactions to familiar communication patterns.
- **Crisis Reset**: During crisis periods, learning may be "reset" as new, unprecedented language appears and uncertainty spikes.

### Empirical Implementation

The extension introduces several alternative similarity measures and time-varying mechanisms:

#### 1. Rolling Window Similarity
- Compares current statement to the **last 5 statements** (instead of just the previous one)
- Captures communication novelty relative to recent history
- Variable: `similarity_rolling5`, `novelty_rolling5 = 1 - similarity_rolling5`

#### 2. Cumulative Similarity Index
- Average similarity to **all past statements**
- Measures overall market familiarity with ECB language patterns
- Variable: `cumulative_similarity`

#### 3. Decay-Weighted Similarity
- Exponentially weighted similarity where **recent statements have higher weight**
- Weight formula: `w_j = exp(-λ * j)` for statement j periods ago
- Decay parameter: λ = 0.5 (half-life decay)
- Variable: `similarity_decay`

#### 4. Regime-Based Analysis
- **Crisis indicator**: Binary variable for major crisis periods (GFC, Eurozone crisis, COVID, Ukraine)
- **Post-Draghi indicator**: After "Whatever it takes" speech (July 26, 2012)
- **Post-Forward Guidance**: After July 2013 policy change
- **Early vs Late period**: Sample split at median date

#### 5. Time-Varying Interactions
- `sim_x_time`: Interaction between similarity and normalized time trend
- `sim_x_crisis`: Interaction between similarity and crisis indicator
- `sim_x_late`: Interaction between similarity and late-period indicator

### Expected Results

| Hypothesis | Expected Sign | Interpretation |
|------------|--------------|----------------|
| Learning effect | sim_x_time < 0 | Market response to similarity diminishes over time |
| Crisis reset | sim_x_crisis > 0 | Similarity matters more during crises |
| Habituation | log_sim_rolling5 < log_sim_jaccard | Rolling measure less predictive as markets adapt |

---

## Extension 2: Risk vs. Uncertainty Decomposition

### Economic Intuition

Following **Knight (1921)**, this extension distinguishes between:
- **Risk**: Situations where probabilities are known or estimable
- **Uncertainty**: Situations where probabilities cannot be estimated (Knightian uncertainty)

This distinction matters for financial markets because:
- **Risk** can be priced and hedged through standard financial instruments
- **Uncertainty** leads to volatility premiums and risk-aversion spikes
- Markets may respond differently to each type of language

The economic literature (e.g., **Baker, Bloom & Davis, 2016**) shows that uncertainty shocks have distinct real effects beyond standard risk measures.

### Empirical Implementation

#### Dictionary Construction

**Risk Dictionary** (~90 terms):
```
risk, risks, probability, likely, expected, forecast, estimate, scenario,
variance, volatility, exposure, downside, hedge, stress, default, credit,
leverage, contagion, systemic, loss, provision, solvency, liquidity...
```

**Uncertainty Dictionary** (~80 terms + Loughran-McDonald uncertainty words):
```
uncertainty, unpredictable, unknown, unclear, ambiguous, vague, indefinite,
unforeseen, unexpected, surprise, shock, unprecedented, unusual, volatile,
turbulence, disruption, instability, turmoil, confusion, complexity, doubt...
```

#### Constructed Variables

| Variable | Formula | Interpretation |
|----------|---------|----------------|
| `risk_index` | risk_terms / n_tokens | Proportion of risk-related language |
| `uncertainty_index` | unc_terms / n_tokens | Proportion of uncertainty-related language |
| `uncertainty_risk_ratio` | unc_index / risk_index | Relative uncertainty vs risk emphasis |
| `pessimism_risk` | (neg_risk - pos_risk) / total | Pessimism from risk words |
| `pessimism_uncertainty` | (neg_unc - pos_unc) / total | Pessimism from uncertainty words |
| `pessimism_other` | Residual | Pessimism from other negative words |

#### Decomposition Validation

The decomposition satisfies:
```
pessimism_lm ≈ pessimism_risk + pessimism_uncertainty + pessimism_other
```

### Expected Results

| Hypothesis | Expected Sign | Interpretation |
|------------|--------------|----------------|
| Uncertainty premium | unc_z > risk_z | Uncertainty affects markets more than risk |
| Differential effects | Different coefficients | Markets distinguish between risk and uncertainty |
| Ratio effect | log_unc_risk_ratio > 0 | Higher uncertainty-to-risk ratio increases volatility |

---

## Implementation Files

### Core Extension Scripts

| Script | Description |
|--------|-------------|
| `src/06_learning_over_time.py` | Extension 1: Computes learning measures |
| `src/04b_risk_uncertainty.py` | Extension 2: Computes risk/uncertainty indices |
| `src/09_extension_regressions.py` | Runs all extension regressions |
| `src/10_extension_figures.py` | Generates publication-ready figures |

### Pipeline Integration

The extensions are fully integrated into `run_pipeline.py` and execute in the correct order:

```
01_scrape_ecb.py          → Raw data collection
02_clean_text.py          → Text preprocessing
03_similarity.py          → Base similarity measures
04_sentiment_lm.py        → LM pessimism score
04b_risk_uncertainty.py   → [NEW] Risk/uncertainty decomposition
05_market_data.py         → Market data download
06_learning_over_time.py  → [NEW] Learning measures
07_event_study_constant_mean.py → Event study
08_regressions.py         → Original regressions
09_extension_regressions.py     → [NEW] Extension regressions
10_extension_figures.py         → [NEW] Publication figures
```

---

## Output Files

### Regression Tables

| File | Content |
|------|---------|
| `outputs/extension1_learning_table.txt` | Learning Over Time regression results |
| `outputs/extension1_learning_table.tex` | LaTeX-exportable version |
| `outputs/extension2_risk_uncertainty_table.txt` | Risk vs Uncertainty regression results |
| `outputs/extension2_risk_uncertainty_table.tex` | LaTeX-exportable version |
| `outputs/extensions_combined_table.txt` | Combined specification results |
| `outputs/extensions_combined_table.tex` | LaTeX-exportable version |

### Data Files

| File | Content |
|------|---------|
| `data/processed/regression_dataset_extended.csv` | Full dataset with all extension variables |

### Figures

| File | Description |
|------|-------------|
| `outputs/figures/fig1_similarity_measures.pdf/png` | Similarity measures over time |
| `outputs/figures/fig2_market_response_evolution.pdf/png` | CAR evolution over time |
| `outputs/figures/fig3_crisis_comparison.pdf/png` | Crisis vs non-crisis comparison |
| `outputs/figures/fig4_risk_uncertainty_indices.pdf/png` | Risk and uncertainty indices |
| `outputs/figures/fig5_risk_uncertainty_scatter.pdf/png` | Risk vs uncertainty scatter |
| `outputs/figures/fig6_decomposed_sentiment.pdf/png` | Sentiment decomposition |
| `outputs/figures/fig7_correlation_heatmap.pdf/png` | Variable correlation matrix |
| `outputs/figures/fig8_summary_statistics.pdf/png` | Summary statistics |

---

## Running the Extensions

### Full Pipeline

```bash
python run_pipeline.py
```

### Individual Extension Scripts

```bash
# Extension 1: Learning Over Time
python src/06_learning_over_time.py

# Extension 2: Risk vs Uncertainty
python src/04b_risk_uncertainty.py

# Extension Regressions (requires both extensions above)
python src/09_extension_regressions.py

# Generate Figures
python src/10_extension_figures.py
```

---

## Regression Specifications

### Extension 1: Learning Over Time Models

| Model | Specification | Key Variables |
|-------|--------------|---------------|
| L1 | Baseline | log_sim_jaccard, pessimism_lm, int_jaccard |
| L2 | Rolling window | log_sim_rolling5, pessimism_lm |
| L3 | Decay-weighted | log_sim_decay, pessimism_lm |
| L4 | Time interaction | + time_trend_norm, sim_x_time |
| L5 | Crisis regime | + crisis, sim_x_crisis, pess_x_crisis |
| L6 | Period split | + late_period, sim_x_late, pess_x_late |

### Extension 2: Risk vs Uncertainty Models

| Model | Specification | Key Variables |
|-------|--------------|---------------|
| U1 | Baseline | log_sim_jaccard, pessimism_lm, int_jaccard |
| U2 | Risk only | log_sim_jaccard, risk_z |
| U3 | Uncertainty only | log_sim_jaccard, unc_z |
| U4 | Horse race | log_sim_jaccard, risk_z, unc_z |
| U5 | Decomposed | pessimism_risk, pessimism_uncertainty, pessimism_other |
| U6 | Interactions | + int_jaccard_risk, int_jaccard_unc |
| U7 | Ratio | + log_unc_risk_ratio |

### Combined Models

| Model | Specification | Description |
|-------|--------------|-------------|
| C1 | Baseline | Original specification |
| C2 | Learning + R/U | Decay similarity + risk/uncertainty decomposition |
| C3 | Full model | + crisis + controls |
| C4 | Kitchen sink | All interactions |

---

## Variable Definitions

### Learning Over Time Variables

| Variable | Definition |
|----------|------------|
| `similarity_rolling5` | Average Jaccard similarity to last 5 statements |
| `novelty_rolling5` | 1 - similarity_rolling5 |
| `cumulative_similarity` | Average similarity to all past statements |
| `similarity_decay` | Exponentially decay-weighted similarity (λ=0.5) |
| `ecb_experience` | Cumulative count of ECB statements |
| `crisis` | Binary indicator for crisis periods |
| `post_draghi` | Post "Whatever it takes" indicator |
| `late_period` | Second half of sample indicator |

### Risk vs Uncertainty Variables

| Variable | Definition |
|----------|------------|
| `risk_index` | Proportion of risk-related terms |
| `uncertainty_index` | Proportion of uncertainty-related terms |
| `risk_z` | Standardized risk index |
| `unc_z` | Standardized uncertainty index |
| `uncertainty_risk_ratio` | uncertainty_index / risk_index |
| `pessimism_risk` | Pessimism contribution from risk words |
| `pessimism_uncertainty` | Pessimism contribution from uncertainty words |
| `pessimism_other` | Residual pessimism |

---

## References

- Baker, S. R., Bloom, N., & Davis, S. J. (2016). Measuring economic policy uncertainty. *Quarterly Journal of Economics*, 131(4), 1593-1636.
- Knight, F. H. (1921). *Risk, Uncertainty and Profit*. Houghton Mifflin.
- Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *Journal of Finance*, 66(1), 35-65.

---

## Authors

ECB Communication Research Project - Extensions
