import os
from pathlib import Path
import pandas as pd
import numpy as np

# --- Force working directory to project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)

# Paths
TEXT_PATH = Path("data/processed/ecb_text_features.csv")
MARKET_PATH = Path("data/market/eurostoxx50_daily.csv")
OUT_PATH = Path("data/processed/event_study_constant_mean.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load data
text_df = pd.read_csv(TEXT_PATH)
market_df = pd.read_csv(MARKET_PATH)

text_df["date"] = pd.to_datetime(text_df["date"])
market_df["date"] = pd.to_datetime(market_df["date"])

market_df = market_df.sort_values("date").reset_index(drop=True)

print("="*80)
print("Text data loaded:", len(text_df), "events")
print("Date range:", text_df["date"].min(), "to", text_df["date"].max())
print("Market data loaded:", len(market_df), "days")
print("Market date range:", market_df["date"].min(), "to", market_df["date"].max())
print("="*80)

WINDOW_EVENT = 5
WINDOW_EST_START = 250
WINDOW_EST_END = 50

results = []

for _, row in text_df.iterrows():
    event_date = row["date"]

    if event_date not in market_df["date"].values:
        continue

    idx = market_df.index[market_df["date"] == event_date][0]

    # Check estimation window availability
    est_start = idx - WINDOW_EST_START
    est_end = idx - WINDOW_EST_END
    evt_start = idx - WINDOW_EVENT
    evt_end = idx + WINDOW_EVENT

    if est_start < 0 or evt_start < 0 or evt_end >= len(market_df):
        continue

    # Estimation window mean return
    mean_return = market_df.loc[est_start:est_end, "return"].mean()

    # Event window abnormal returns
    event_returns = market_df.loc[evt_start:evt_end, "return"]
    abnormal_returns = event_returns - mean_return

    car = abnormal_returns.sum()
    abs_car = abs(car)

    results.append({
        "date": event_date,
        "CAR_m5_p5": car,
        "abs_CAR_m5_p5": abs_car
    })

event_df = pd.DataFrame(results)

# Merge with text variables
final_df = event_df.merge(
    text_df,
    on="date",
    how="left"
)

final_df = final_df.sort_values("date").reset_index(drop=True)

# Add year column for easy filtering
final_df["year"] = final_df["date"].dt.year

final_df.to_csv(OUT_PATH, index=False)

print("="*80)
print("EVENT STUDY RESULTS")
print("="*80)
print(final_df[["date", "CAR_m5_p5", "abs_CAR_m5_p5", "year"]].head(10))
print("...")
print(final_df[["date", "CAR_m5_p5", "abs_CAR_m5_p5", "year"]].tail(10))
print("="*80)
print("Total events with CAR:", len(final_df))
print("Date range:", final_df["date"].min(), "to", final_df["date"].max())
print("Year range:", final_df["year"].min(), "to", final_df["year"].max())
print("="*80)
print("Saved:", OUT_PATH.resolve())
print("="*80)
