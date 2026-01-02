import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.util import bigrams

# --- Force working directory to project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)

IN_PATH = Path("data/processed/ecb_text_features.csv")
OUT_PATH = Path("data/processed/ecb_text_features.csv")  # overwrite same file

df = pd.read_csv(IN_PATH)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# -------------------------
# 1) JACCARD SIMILARITY (BIGRAMS) — REPLICATION
# -------------------------
def jaccard_similarity(tokens1, tokens2):
    if not isinstance(tokens1, list) or not isinstance(tokens2, list):
        return np.nan
    b1 = set(bigrams(tokens1))
    b2 = set(bigrams(tokens2))
    if len(b1 | b2) == 0:
        return np.nan
    return len(b1 & b2) / len(b1 | b2)

jaccard = [np.nan]
for i in range(1, len(df)):
    t1 = eval(df.loc[i-1, "tokens"])
    t2 = eval(df.loc[i, "tokens"])
    jaccard.append(jaccard_similarity(t1, t2))

df["similarity_jaccard"] = jaccard

# -------------------------
# 2) COSINE SIMILARITY (BoW) — EXTENSION
# -------------------------
texts = df["tokens"].apply(lambda x: " ".join(eval(x))).tolist()

cv = CountVectorizer(min_df=5)
X = cv.fit_transform(texts).toarray()

def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return np.nan
    return np.dot(a, b) / denom

cosine = [np.nan]
for i in range(1, len(X)):
    cosine.append(cosine_sim(X[i-1], X[i]))

df["similarity_cosine"] = cosine

# -------------------------
# Save
# -------------------------
df.to_csv(OUT_PATH, index=False)

print(df[["date", "similarity_jaccard", "similarity_cosine"]].head())
print(df[["date", "similarity_jaccard", "similarity_cosine"]].tail())
print("Done. Similarity measures added.")
