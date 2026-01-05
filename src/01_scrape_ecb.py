import os
from pathlib import Path
from selenium import webdriver
import time
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import pandas as pd

# --- Force working directory to project root ---
ROOT_DIR = Path(__file__).resolve().parent.parent
os.chdir(ROOT_DIR)

BASE = "https://www.ecb.europa.eu"
INDEX_URL = f"{BASE}/press/pressconf/html/index.en.html"

OUT_PATH = Path("data/raw/ecb_press_conferences_raw.csv")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# 1) Open index page with Selenium (loads more links via scrolling)
driver = webdriver.Firefox()  # or webdriver.Chrome()
driver.get(INDEX_URL)

for y in range(0, 12000, 300):
    driver.execute_script(f"window.scrollBy(0, {y});")
    time.sleep(0.4)

html = driver.page_source
driver.quit()

soup_index = BeautifulSoup(html, "html.parser")

# 2) Collect links
links = []
for a in soup_index.find_all("a", href=True):
    href = a["href"]
    if ("press_conference/monetary-policy" in href) and href.endswith(".en.html"):
        links.append(href)

links = sorted(list(set(links)))
print(f"Found {len(links)} press conference pages.")

# 3) Visit each page and extract date + intro statement text
rows = []
headers = {"User-Agent": "Mozilla/5.0"}

for href in tqdm(links):
    url = BASE + href
    req = requests.get(url, headers=headers, timeout=30)
    req.raise_for_status()
    soup = BeautifulSoup(req.content, "html.parser")

    # date (meta tag)
    date_val = None
    meta = soup.find("meta", {"property": "article:published_time"})
    if meta and meta.get("content"):
        date_val = meta["content"]

    # content: take paragraphs/headings inside main "section" blocks
    parts = []
    for section in soup.find_all("div", {"class": "section"}):
        for tag in section.find_all(["h2", "p"]):
            txt = tag.get_text(" ", strip=True)
            if txt:
                parts.append(txt)

    content_raw = "\n".join(parts).strip()

    # Keep only non-empty
    if date_val and content_raw:
        rows.append({"date": date_val, "content_raw": content_raw, "url": url})

df = pd.DataFrame(rows)
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

print(df.head(3))
print(df.tail(3))
print("Rows:", len(df))

    # Save
df.to_csv(OUT_PATH, index=False)
print(f"Saved: {OUT_PATH.resolve()}")
