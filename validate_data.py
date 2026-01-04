"""
Quick validation script to check data availability and structure.
Run this before executing the full pipeline to ensure all data is present.
"""

import os
from pathlib import Path
import pandas as pd

# Force working directory to project root
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

print("="*80)
print("COMPLETE DATA VALIDATION CHECK")
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
check_file(Path("data/processed/ecb_text_features.csv"), "Text Features (base)")
check_file(Path("data/processed/event_study_constant_mean.csv"), "Event Study Results")
check_file(Path("data/processed/regression_dataset.csv"), "Regression Dataset (base)")
check_file(Path("data/processed/regression_dataset_extended.csv"), "Regression Dataset (extended with both extensions)")

# Check market data
print("MARKET DATA:")
print("-"*80)
check_file(Path("data/market/eurostoxx50_daily.csv"), "Eurostoxx50 Daily")

# Check outputs
print("OUTPUT FILES:")
print("-"*80)
print("Core Replication:")
check_file(Path("outputs/replication_table.txt"), "  Replication Table (2007-2013)")
check_file(Path("outputs/extension_table.txt"), "  Extension Table (2007-2025)")
print("\nExtension 1 - Learning Over Time:")
check_file(Path("outputs/extension1_learning_table.txt"), "  Learning Table (txt)")
check_file(Path("outputs/extension1_learning_table.tex"), "  Learning Table (LaTeX)")
print("\nExtension 2 - Risk vs Uncertainty:")
check_file(Path("outputs/extension2_risk_uncertainty_table.txt"), "  Risk/Uncertainty Table (txt)")
check_file(Path("outputs/extension2_risk_uncertainty_table.tex"), "  Risk/Uncertainty Table (LaTeX)")
print("\nCombined Extensions:")
check_file(Path("outputs/extensions_combined_table.txt"), "  Combined Table (txt)")
check_file(Path("outputs/extensions_combined_table.tex"), "  Combined Table (LaTeX)")

# Check figures directory
print("\nFIGURES:")
print("-"*80)
figures_dir = Path("outputs/figures")
if figures_dir.exists():
    figures = list(figures_dir.glob("*.pdf"))
    print(f"✓ Figures directory exists")
    print(f"  Found {len(figures)} PDF figures")
    if figures:
        for fig in sorted(figures)[:8]:  # Show first 8
            print(f"    - {fig.name}")
else:
    print(f"✗ Figures directory not found")
print()

print("="*80)
print("VALIDATION COMPLETE")
print("="*80)

# Summary
raw_exists = Path("data/raw/ecb_press_conferences_raw.csv").exists()
lm_exists = Path("data/raw/Loughran-McDonald_MasterDictionary_1993-2024.csv").exists()
processed_exists = Path("data/processed/regression_dataset.csv").exists()
extended_exists = Path("data/processed/regression_dataset_extended.csv").exists()

print("\nPIPELINE STATUS:")
print("-"*80)
if raw_exists and lm_exists:
    print("✓ Raw data present - ready to run pipeline")
    if processed_exists:
        print("✓ Base processed data exists")
    if extended_exists:
        print("✓ Extended processed data exists (with extensions)")
    if not processed_exists:
        print("⚠ Need to run pipeline to generate processed data")
        print("  Run: python run_pipeline.py")
elif not raw_exists and not lm_exists:
    print("⚠ Raw data missing - need to run scraping and download LM dictionary")
    print("  1. Run: python src/01_scrape_ecb.py")
    print("  2. Download LM dictionary from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/")
elif not raw_exists:
    print("⚠ ECB data missing - need to run scraping")
    print("  Run: python src/01_scrape_ecb.py")
elif not lm_exists:
    print("⚠ LM dictionary missing")
    print("  Download from: https://sraf.nd.edu/loughranmcdonald-master-dictionary/")

# Dataset statistics
if processed_exists:
    print("\nDATASET STATISTICS:")
    print("-"*80)
    df = pd.read_csv("data/processed/regression_dataset.csv")
    print(f"Total observations: {len(df)}")
    print(f"Total variables: {len(df.columns)}")
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        repl_df = df[df['date'] <= '2013-12-31']
        ext_df = df[df['date'] > '2013-12-31']
        print(f"Replication sample (≤2013): {len(repl_df)} observations")
        print(f"Extension sample (>2013): {len(ext_df)} observations")
    
    # Check for extension variables
    ext1_vars = ['similarity_rolling5', 'cumulative_similarity', 'similarity_decay', 'ecb_experience']
    ext2_vars = ['risk_index', 'uncertainty_index', 'pessimism_risk', 'pessimism_uncertainty']
    
    has_ext1 = all(var in df.columns for var in ext1_vars)
    has_ext2 = all(var in df.columns for var in ext2_vars)
    
    print(f"\nExtension 1 (Learning Over Time) variables: {'✓ Present' if has_ext1 else '✗ Missing'}")
    print(f"Extension 2 (Risk vs Uncertainty) variables: {'✓ Present' if has_ext2 else '✗ Missing'}")

print("="*80)
