"""
Quick validation script to check data availability and structure.
Run this before executing the full pipeline.
"""

import os
from pathlib import Path
import pandas as pd

# Force working directory to project root
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

print("="*80)
print("DATA VALIDATION CHECK")
print("="*80)
print(f"Working directory: {ROOT_DIR}")
print("="*80)

def check_file(path, description):
    """Check if a file exists and show basic info."""
    if path.exists():
        if path.suffix == '.csv':
            try:
                df = pd.read_csv(path)
                print(f"✓ {description}")
                print(f"  Path: {path}")
                print(f"  Rows: {len(df):,}")
                print(f"  Columns: {list(df.columns)}")
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
            except Exception as e:
                print(f"⚠ {description} - Error reading file: {e}")
                print(f"  Path: {path}")
        else:
            size = path.stat().st_size / 1024
            print(f"✓ {description}")
            print(f"  Path: {path}")
            print(f"  Size: {size:.1f} KB")
    else:
        print(f"✗ {description} - FILE NOT FOUND")
        print(f"  Expected path: {path}")
    print()

# Check raw data
print("RAW DATA:")
print("-"*80)
check_file(Path("data/raw/ecb_press_conferences_raw.csv"), "ECB Press Conferences (raw)")
check_file(Path("data/raw/Loughran-McDonald_MasterDictionary_1993-2024.csv"), "Loughran-McDonald Dictionary")

# Check processed data
print("PROCESSED DATA:")
print("-"*80)
check_file(Path("data/processed/ecb_text_features.csv"), "Text Features")
check_file(Path("data/processed/event_study_constant_mean.csv"), "Event Study Results")
check_file(Path("data/processed/regression_dataset.csv"), "Regression Dataset")

# Check market data
print("MARKET DATA:")
print("-"*80)
check_file(Path("data/market/eurostoxx50_daily.csv"), "Eurostoxx50 Daily")

# Check outputs
print("OUTPUT FILES:")
print("-"*80)
check_file(Path("outputs/replication_table.txt"), "Replication Table (2007-2013)")
check_file(Path("outputs/extension_table.txt"), "Extension Table (2007-2025)")

print("="*80)
print("VALIDATION COMPLETE")
print("="*80)

# Summary
raw_exists = Path("data/raw/ecb_press_conferences_raw.csv").exists()
lm_exists = Path("data/raw/Loughran-McDonald_MasterDictionary_1993-2024.csv").exists()

if raw_exists and lm_exists:
    print("✓ Raw data present - ready to run pipeline")
elif not raw_exists and not lm_exists:
    print("⚠ Raw data missing - need to run scraping and download LM dictionary")
    print("  Run: python src/01_scrape_ecb.py")
    print("  Download LM dictionary from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/")
elif not raw_exists:
    print("⚠ ECB data missing - need to run scraping")
    print("  Run: python src/01_scrape_ecb.py")
elif not lm_exists:
    print("⚠ LM dictionary missing")
    print("  Download from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/")

print("="*80)
