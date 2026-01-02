# Regression Specifications

## Replication Benchmark (2007-2013)

**Sample:** Overlap period between original paper (1999-2013) and Yahoo Finance data availability (2007+)

### Model R1: Baseline Replication
```
abs_CAR = β₀ + β₁·log(similarity_jaccard) + β₂·pessimism_lm + β₃·[log(similarity_jaccard) × pessimism_lm] + ε
```

**Variables:**
- `abs_CAR_m5_p5`: Absolute cumulative abnormal return over [-5, +5] days
- `log_sim_jaccard`: Log of Jaccard bigram similarity with previous statement
- `pessimism_lm`: Loughran-McDonald pessimism = (negative - positive) / (negative + positive)
- `int_jaccard`: Interaction term

**Expected Results:**
- β₁ < 0: Higher similarity → lower market volatility
- β₂ > 0: More pessimistic tone → higher market volatility
- β₃: Interaction effect (dampening or amplifying)

### Model R2: With Controls
```
abs_CAR = β₀ + β₁·log(similarity) + β₂·pessimism + β₃·interaction 
          + β₄·n_tokens + β₅·time + ε
```

**Additional controls:**
- `n_tokens`: Document length (number of tokens after preprocessing)
- `time`: Time trend (days since first observation)

---

## Extension (2007-2025)

**Sample:** Full available period

### Model E1: Jaccard Baseline (Extended Period)
Same as R1 but with extended sample through 2025

### Model E2: Jaccard + Controls
Same as R2 but with extended sample through 2025

### Model E3: Cosine Baseline (Alternative Similarity)
```
abs_CAR = β₀ + β₁·log(similarity_cosine) + β₂·pessimism_lm + β₃·[log(similarity_cosine) × pessimism_lm] + ε
```

**Robustness check:** Uses cosine similarity instead of Jaccard
- Cosine similarity computed on BoW vectors (min_df=5)
- Tests whether results are robust to similarity measure choice

### Model E4: Cosine + Controls
Full specification with cosine similarity and all controls

---

## Estimation Details

**Standard Errors:** Heteroskedasticity-robust (HC1)

**Missing Values:** Dropped automatically via `missing="drop"`

**Constant Term:** Included in all models

**Sample Restrictions:**
- Event must have sufficient market data for estimation window [-250, -50]
- Event must have sufficient market data for event window [-5, +5]
- All text features must be non-missing

---

## Expected Output Structure

### Replication Table (outputs/replication_table.txt)
```
===========================================
  REPLICATION BENCHMARK: 2007-2013
===========================================
                  Baseline  With Controls
-------------------------------------------
const               [...]        [...]
log_sim_jaccard     [...]        [...]
pessimism_lm        [...]        [...]
int_jaccard         [...]        [...]
n_tokens                         [...]
time                             [...]
-------------------------------------------
N                    XX           XX
R²                  0.XXX        0.XXX
Adj. R²             0.XXX        0.XXX
===========================================
```

### Extension Table (outputs/extension_table.txt)
```
===============================================================
  EXTENSION: 2007-2025
===============================================================
            Jaccard  Jaccard+Ctrl  Cosine  Cosine+Ctrl
---------------------------------------------------------------
const        [...]      [...]      [...]     [...]
log_sim_*    [...]      [...]      [...]     [...]
pessimism_lm [...]      [...]      [...]     [...]
int_*        [...]      [...]      [...]     [...]
n_tokens                [...]                [...]
time                    [...]                [...]
---------------------------------------------------------------
N             XXX        XXX        XXX       XXX
R²           0.XXX      0.XXX      0.XXX     0.XXX
Adj. R²      0.XXX      0.XXX      0.XXX     0.XXX
===============================================================
```

---

## Interpretation Guide

**Significant negative β₁ (log_similarity):**
- Indicates that higher similarity to previous statement reduces market volatility
- Suggests markets react less when communication is predictable

**Significant positive β₂ (pessimism):**
- Indicates that pessimistic tone increases market volatility
- Consistent with sentiment-driven market reactions

**Significant interaction β₃:**
- Positive: Pessimism amplifies similarity effect
- Negative: Pessimism dampens similarity effect

**Controls:**
- `n_tokens`: Controls for document length/information quantity
- `time`: Controls for structural changes over time

**Robustness (Cosine vs Jaccard):**
- If E1 ≈ E3 and E2 ≈ E4: Results are robust to similarity measure
- If different: Results depend on how similarity is measured
