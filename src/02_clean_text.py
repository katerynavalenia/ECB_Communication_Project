import os
from pathlib import Path
import pandas as pd
import numpy as np
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# --- Force working directory to project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)

# If you get a missing resource error, uncomment these once:
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("omw-1.4")

IN_PATH = Path("data/raw/ecb_press_conferences_raw.csv")
OUT_PATH = Path("data/processed/ecb_text_features.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

stops = set(stopwords.words("english"))
wn = WordNetLemmatizer()

def clean_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stops and len(t) > 1]
    tokens = [wn.lemmatize(t) for t in tokens]
    return tokens

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

df["tokens"] = df["content_raw"].apply(clean_text)
df["n_tokens"] = df["tokens"].apply(len)

# quick sanity checks
print(df[["date", "n_tokens"]].head())
print(df[["date", "n_tokens"]].tail())
print("Rows:", len(df), "Avg tokens:", int(df["n_tokens"].mean()))

df.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH.resolve()}")
