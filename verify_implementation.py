"""
Test script to verify the replication-first workflow implementation.
Checks that all paths, filtering, and outputs are correctly configured.
"""

import os
from pathlib import Path

# Force working directory to project root
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

print("="*80)
print("REPLICATION-FIRST WORKFLOW VERIFICATION")
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
    "src/05_market_data.py",
    "src/07_event_study_constant_mean.py",
    "src/08_regressions.py"
]

for file_path in source_files:
    path = Path(file_path)
    exists = "✓" if path.exists() else "✗"
    
    # Check for ROOT_DIR pattern in files 07 and 08
    if path.exists() and file_path in ["src/07_event_study_constant_mean.py", "src/08_regressions.py"]:
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

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
print("\nIf all checks show ✓, the replication-first workflow is correctly implemented.")
print("Run 'python run_pipeline.py' to execute the full pipeline.")
print("="*80)
