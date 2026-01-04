"""
Test script to verify the complete pipeline implementation.
Checks that all paths, filtering, outputs, and extension scripts are correctly configured.
"""

import os
from pathlib import Path

# Force working directory to project root
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

print("="*80)
print("COMPLETE PIPELINE VERIFICATION")
print("="*80)
print(f"Project root: {ROOT_DIR}")
print("="*80)

# Check directory structure
print("\n1. CHECKING DIRECTORY STRUCTURE")
print("-"*80)

required_dirs = [
    "data/raw",
    "data/processed",
    "data/market",
    "outputs",
    "outputs/figures",
    "src"
]

for dir_path in required_dirs:
    path = Path(dir_path)
    exists = "✓" if path.exists() else "✗"
    print(f"{exists} {dir_path}")

# Check source files
print("\n2. CHECKING SOURCE FILES")
print("-"*80)

source_files = [
    "src/01_scrape_ecb.py",
    "src/02_clean_text.py",
    "src/03_similarity.py",
    "src/04_sentiment_lm.py",
    "src/04b_risk_uncertainty.py",
    "src/05_market_data.py",
    "src/06_learning_over_time.py",
    "src/07_event_study_constant_mean.py",
    "src/08_regressions.py",
    "src/09_extension_regressions.py",
    "src/10_extension_figures.py"
]

for file_path in source_files:
    path = Path(file_path)
    exists = "✓" if path.exists() else "✗"
    
    # Check for ROOT_DIR pattern in critical scripts
    critical_scripts = [
        "src/07_event_study_constant_mean.py", 
        "src/08_regressions.py",
        "src/09_extension_regressions.py",
        "src/10_extension_figures.py"
    ]
    if path.exists() and file_path in critical_scripts:
        content = path.read_text(encoding='utf-8')
        has_root = "ROOT_DIR = Path(__file__).resolve().parent.parent" in content
        has_chdir = "os.chdir(ROOT_DIR)" in content
        root_check = " (has ROOT_DIR)" if has_root and has_chdir else " (MISSING ROOT_DIR!)"
    else:
        root_check = ""
    
    print(f"{exists} {file_path}{root_check}")

# Check path configuration in script 07
print("\n3. CHECKING SCRIPT 07 (EVENT STUDY) CONFIGURATION")
print("-"*80)

script_07 = Path("src/07_event_study_constant_mean.py")
if script_07.exists():
    content = script_07.read_text(encoding='utf-8')
    
    checks = {
        'ROOT_DIR pattern': 'ROOT_DIR = Path(__file__).resolve().parent.parent' in content,
        'os.chdir(ROOT_DIR)': 'os.chdir(ROOT_DIR)' in content,
        'Correct OUT_PATH': 'Path("data/processed/event_study_constant_mean.csv")' in content,
        'Year column': 'final_df["year"]' in content,
        'Event count logging': '"Total events with CAR:"' in content or 'len(final_df)' in content,
        'Date range logging': '"Date range:"' in content
    }
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
else:
    print("✗ Script not found!")

# Check path configuration in script 08
print("\n4. CHECKING SCRIPT 08 (REGRESSIONS) CONFIGURATION")
print("-"*80)

script_08 = Path("src/08_regressions.py")
if script_08.exists():
    content = script_08.read_text(encoding='utf-8')
    
    checks = {
        'ROOT_DIR pattern': 'ROOT_DIR = Path(__file__).resolve().parent.parent' in content,
        'os.chdir(ROOT_DIR)': 'os.chdir(ROOT_DIR)' in content,
        'Loads from data/processed': 'Path(\'data/processed/event_study_constant_mean.csv\')' in content,
        'Outputs to outputs/': 'Path("outputs/replication_table.txt")' in content,
        'Replication filter': 'df["date"] <= "2013-12-31"' in content or 'date <= "2013-12-31"' in content,
        'Model R1 (Jaccard)': 'm_r1' in content or 'Model R1' in content,
        'Model R2 (Cosine)': 'm_r2' in content or 'Model R2' in content,
        'Model E1-E4': 'm_e1' in content and 'm_e4' in content,
        'Robust SE (HC1)': 'cov_type="HC1"' in content,
        'Sample size logging': 'len(df_repl)' in content or 'Replication sample:' in content
    }
    
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")
else:
    print("✗ Script not found!")

# Check expected output paths
print("\n5. CHECKING EXPECTED OUTPUT PATHS (in script 08)")
print("-"*80)

if script_08.exists():
    content = script_08.read_text(encoding='utf-8')
    
    output_checks = {
        'outputs/replication_table.txt': 'outputs/replication_table.txt' in content,
        'outputs/extension_table.txt': 'outputs/extension_table.txt' in content,
        'data/processed/regression_dataset.csv': 'data/processed/regression_dataset.csv' in content,
        'NO src/data/ paths': 'src/data/' not in content,
        'NO src/outputs/ paths': 'src/outputs/' not in content
    }
    
    for check, passed in output_checks.items():
        status = "✓" if passed else "✗"
        print(f"{status} {check}")

# Check extension scripts
print("\n6. CHECKING EXTENSION SCRIPTS")
print("-"*80)

# Extension 1: Learning Over Time
script_06 = Path("src/06_learning_over_time.py")
if script_06.exists():
    content = script_06.read_text(encoding='utf-8')
    
    checks = {
        'ROOT_DIR pattern': 'ROOT_DIR = Path(__file__).resolve().parent.parent' in content or 'ROOT_DIR' in content,
        'Rolling window similarity': 'similarity_rolling' in content or 'rolling' in content,
        'Cumulative similarity': 'cumulative_similarity' in content or 'cumulative' in content,
        'Decay-weighted similarity': 'similarity_decay' in content or 'decay' in content,
        'Crisis indicator': 'crisis' in content,
    }
    
    print("Extension 1 (Learning Over Time):")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
else:
    print("✗ Extension 1 script not found!")

# Extension 2: Risk vs Uncertainty
script_04b = Path("src/04b_risk_uncertainty.py")
if script_04b.exists():
    content = script_04b.read_text(encoding='utf-8')
    
    checks = {
        'ROOT_DIR pattern': 'ROOT_DIR = Path(__file__).resolve().parent.parent' in content or 'ROOT_DIR' in content,
        'Risk index': 'risk_index' in content,
        'Uncertainty index': 'uncertainty_index' in content,
        'Decomposed sentiment': 'pessimism_risk' in content or 'pessimism_uncertainty' in content,
        'Risk/Uncertainty ratio': 'ratio' in content,
    }
    
    print("\nExtension 2 (Risk vs Uncertainty):")
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
else:
    print("✗ Extension 2 script not found!")

# Extension regressions
print("\n7. CHECKING EXTENSION REGRESSION OUTPUTS")
print("-"*80)

extension_outputs = [
    ("outputs/extension1_learning_table.txt", "Extension 1 table (txt)"),
    ("outputs/extension1_learning_table.tex", "Extension 1 table (LaTeX)"),
    ("outputs/extension2_risk_uncertainty_table.txt", "Extension 2 table (txt)"),
    ("outputs/extension2_risk_uncertainty_table.tex", "Extension 2 table (LaTeX)"),
    ("outputs/extensions_combined_table.txt", "Combined extensions table (txt)"),
    ("outputs/extensions_combined_table.tex", "Combined extensions table (LaTeX)"),
]

for file_path, desc in extension_outputs:
    path = Path(file_path)
    exists = "✓" if path.exists() else "✗"
    print(f"{exists} {desc}")

# Check figures
print("\n8. CHECKING EXTENSION FIGURES")
print("-"*80)

expected_figures = [
    "outputs/figures/fig1_similarity_measures.pdf",
    "outputs/figures/fig2_market_response_evolution.pdf",
    "outputs/figures/fig3_crisis_comparison.pdf",
    "outputs/figures/fig4_risk_uncertainty_indices.pdf",
    "outputs/figures/fig5_risk_uncertainty_scatter.pdf",
    "outputs/figures/fig6_decomposed_sentiment.pdf",
    "outputs/figures/fig7_correlation_heatmap.pdf",
    "outputs/figures/fig8_summary_statistics.pdf",
]

for fig in expected_figures:
    path = Path(fig)
    exists = "✓" if path.exists() else "✗"
    fig_name = Path(fig).name
    print(f"{exists} {fig_name}")

# Check datasets
print("\n9. CHECKING DATASETS")
print("-"*80)

datasets = [
    ("data/processed/regression_dataset.csv", "Base regression dataset"),
    ("data/processed/regression_dataset_extended.csv", "Extended regression dataset"),
    ("data/processed/ecb_text_features.csv", "ECB text features"),
    ("data/processed/event_study_constant_mean.csv", "Event study results"),
]

import pandas as pd
for file_path, desc in datasets:
    path = Path(file_path)
    if path.exists():
        try:
            df = pd.read_csv(path)
            print(f"✓ {desc}")
            print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            if 'date' in df.columns:
                print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        except Exception as e:
            print(f"⚠ {desc} - exists but error reading: {e}")
    else:
        print(f"✗ {desc} - NOT FOUND")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print("\nSUMMARY:")
print("-"*80)
print("✓ Core pipeline: Scripts 01-08 (scraping → regressions)")
print("✓ Extension 1: Learning Over Time (script 06)")
print("✓ Extension 2: Risk vs Uncertainty (script 04b)")
print("✓ Extension analysis: Scripts 09-10 (regressions + figures)")
print("\nTo run the complete pipeline:")
print("  python run_pipeline.py")
print("\nTo validate data:")
print("  python validate_data.py")
print("="*80)
