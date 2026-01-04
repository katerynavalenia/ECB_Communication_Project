import os
from pathlib import Path
import yfinance as yf
import pandas as pd
import numpy as np

# --- Force working directory to project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)

OUT_PATH = Path("data/market/eurostoxx50_daily.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

ticker = "^STOXX50E"

print(f"Downloading {ticker} data from Yahoo Finance...")

df = yf.download(
    ticker,
    start="1998-01-01",
    progress=True,
    auto_adjust=True
)

# Check if download succeeded
if df.empty:
    print("WARNING: yfinance returned empty DataFrame. Trying alternative method...")
    # Try with Ticker object
    tkr = yf.Ticker(ticker)
    df = tkr.history(start="1998-01-01", auto_adjust=True)
    
if df.empty:
    raise ValueError(f"Failed to download data for {ticker}. Check internet connection or try again later.")

print(f"Downloaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Handle multi-level columns from newer yfinance versions
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

# Handle different column name formats (Date vs date, etc.)
df.columns = [str(c).lower() for c in df.columns]

print(f"After processing, columns: {df.columns.tolist()}")

# With auto_adjust=True, "Close" is already adjusted
if "close" in df.columns:
    df = df[["date", "close"]].rename(columns={"close": "price"})
elif "adj close" in df.columns:
    df = df[["date", "adj close"]].rename(columns={"adj close": "price"})
else:
    raise ValueError(f"Could not find Close column. Columns: {df.columns.tolist()}")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ✅ log returns (as in the paper)
df["return"] = np.log(df["price"]).diff()

df = df.dropna()

df.to_csv(OUT_PATH, index=False)

print(df.head())
print(df.tail())
print("Saved:", OUT_PATH.resolve())
print("Start date:", df["date"].min(), "End date:", df["date"].max(), "Rows:", len(df))
