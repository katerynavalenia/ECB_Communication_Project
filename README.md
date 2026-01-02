# ECB Communication Project - Replication & Extension

This project replicates and extends the analysis of ECB press conference communications and their impact on market returns.

## Project Structure

```
ECB_Communication_Project/
├── data/
│   ├── raw/                              # Raw scraped data
│   │   ├── ecb_press_conferences_raw.csv
│   │   └── Loughran-McDonald_MasterDictionary_1993-2024.csv
│   ├── processed/                        # Processed datasets
│   │   ├── ecb_text_features.csv
│   │   ├── event_study_constant_mean.csv
│   │   └── regression_dataset.csv
│   └── market/                           # Market data
│       └── eurostoxx50_daily.csv
├── outputs/                              # Regression tables and results
│   ├── replication_table.txt
│   └── extension_table.txt
├── src/                                  # Source code
│   ├── 01_scrape_ecb.py
│   ├── 02_clean_text.py
│   ├── 03_similarity.py
│   ├── 04_sentiment_lm.py
│   ├── 05_market_data.py
│   ├── 07_event_study_constant_mean.py
│   └── 08_regressions.py
└── run_pipeline.py                       # Master script to run entire pipeline
```

## Pipeline Overview

1. **01_scrape_ecb.py**: Scrapes ECB press conference introductory statements from ECB website
2. **02_clean_text.py**: Cleans and tokenizes text, removes stopwords, lemmatizes
3. **03_similarity.py**: Computes similarity measures (Jaccard bigrams + Cosine)
4. **04_sentiment_lm.py**: Computes Loughran-McDonald pessimism score
5. **05_market_data.py**: Downloads Eurostoxx50 daily prices from Yahoo Finance
6. **07_event_study_constant_mean.py**: Computes CAR(-5,+5) using constant mean return model
7. **08_regressions.py**: Runs replication (2007-2013) and extension (2007-2025) regressions

## Requirements

```bash
pip install pandas numpy scipy scikit-learn statsmodels yfinance nltk selenium beautifulsoup4 requests tqdm
```

NLTK resources (run once):
```python
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")
```

## Usage

### Run Entire Pipeline

```bash
python run_pipeline.py
```

### Run Individual Scripts

```bash
cd ECB_Communication_Project
python src/01_scrape_ecb.py
python src/02_clean_text.py
python src/03_similarity.py
python src/04_sentiment_lm.py
python src/05_market_data.py
python src/07_event_study_constant_mean.py
python src/08_regressions.py
```

## Methodology

### Replication Benchmark (2007-2013)

- **Sample Period**: 2007-2013 (overlap between paper period 1999-2013 and Yahoo Finance data availability starting 2007)
- **Dependent Variable**: |CAR(-5,+5)| (Absolute Cumulative Abnormal Return)
- **Independent Variables**:
  - `log_sim_jaccard`: Log of Jaccard bigram similarity with previous statement
  - `pessimism_lm`: Loughran-McDonald pessimism measure (neg-pos)/(neg+pos)
  - `int_jaccard`: Interaction term (log_sim_jaccard × pessimism_lm)
- **Models**:
  - **R1 (Baseline)**: As in original paper
  - **R2 (With Controls)**: + n_tokens + time trend

### Extension (2007-2025)

- **Sample Period**: 2007-2025 (full available data)
- **Models**:
  - **E1 (Jaccard Baseline)**: Same as R1 but extended period
  - **E2 (Jaccard + Controls)**: + n_tokens + time trend
  - **E3 (Cosine Baseline)**: Alternative similarity measure
  - **E4 (Cosine + Controls)**: + n_tokens + time trend

### Event Study Design

- **Event Window**: [-5, +5] trading days around ECB press conference
- **Estimation Window**: [-250, -50] trading days
- **Model**: Constant mean return model
- **Abnormal Return**: $AR_t = R_t - \bar{R}_{estimation}$
- **CAR**: Sum of abnormal returns over event window

## Output Files

- **outputs/replication_table.txt**: Regression results for 2007-2013 period
- **outputs/extension_table.txt**: Regression results for 2007-2025 period with robustness checks
- **data/processed/regression_dataset.csv**: Full dataset with all variables
- **data/processed/event_study_constant_mean.csv**: Event study results with CAR

## Key Features

- ✅ All scripts force working directory to project root for reproducibility
- ✅ Paths use `data/` and `outputs/` (not `src/data/`)
- ✅ Minimal logging: prints row counts and date ranges
- ✅ Deterministic and reproducible
- ✅ Robust standard errors (HC1) in all regressions
- ✅ Clear separation between replication and extension

## Notes

- Yahoo Finance data for Eurostoxx50 starts in 2007, limiting replication period
- Original paper covered 1999-2013 with proprietary data
- Extension explores robustness with cosine similarity and additional controls
- All text processing follows standard NLP pipeline (lowercase, remove stopwords, lemmatization)

## Authors

ECB Communication Research Project - Replication and Extension Study
