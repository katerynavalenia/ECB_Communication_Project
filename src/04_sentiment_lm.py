import os
from pathlib import Path
import pandas as pd
from nltk.tokenize import word_tokenize

# --- Force working directory to project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)

# Paths
IN_PATH = Path("data/processed/ecb_text_features.csv")
LM_PATH = Path("data/raw/Loughran-McDonald_MasterDictionary_1993-2024.csv")
OUT_PATH = IN_PATH  # overwrite same file

# Load data
df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])

# Load LM dictionary
lm = pd.read_csv(LM_PATH)
neg_words = lm.loc[lm["Negative"] > 0, "Word"].str.lower().tolist()
pos_words = lm.loc[lm["Positive"] > 0, "Word"].str.lower().tolist()

neg_set = set(neg_words)
pos_set = set(pos_words)

def lm_pessimism(text):
    if not isinstance(text, str):
        return 0.0
    tokens = word_tokenize(text.lower())
    pos = sum(t in pos_set for t in tokens)
    neg = sum(t in neg_set for t in tokens)
    total = pos + neg
    if total == 0:
        return 0.0
    return (neg - pos) / total

# Compute sentiment
df["pessimism_lm"] = df["content_raw"].apply(lm_pessimism)

# Save
df.to_csv(OUT_PATH, index=False)

print(df[["date", "pessimism_lm"]].head())
print(df[["date", "pessimism_lm"]].tail())
print("Done. LM sentiment added.")
