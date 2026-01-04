"""
Master script to run the entire ECB Communication Project pipeline.
Executes all scripts in sequence from 01 to 08.
"""

import os
import sys
from pathlib import Path
import subprocess

# Force working directory to project root
ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)

print("="*80)
print("ECB COMMUNICATION PROJECT - MASTER PIPELINE")
print("="*80)
print(f"Working directory: {ROOT_DIR}")
print("="*80)

# Define pipeline scripts in order
SCRIPTS = [
    "src/01_scrape_ecb.py",
    "src/02_clean_text.py",
    "src/03_similarity.py",
    "src/04_sentiment_lm.py",
    "src/04b_risk_uncertainty.py",      # Extension 2: Risk vs Uncertainty
    "src/05_market_data.py",
    "src/06_learning_over_time.py",     # Extension 1: Learning Over Time
    "src/07_event_study_constant_mean.py",
    "src/08_regressions.py",
    "src/09_extension_regressions.py",  # Extension regressions
    "src/10_extension_figures.py",      # Extension figures
]

def run_script(script_path):
    """Run a Python script and handle errors."""
    print("\n" + "="*80)
    print(f"RUNNING: {script_path}")
    print("="*80)
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            cwd=ROOT_DIR,
            capture_output=False  # Show output in real-time
        )
        print(f"✓ SUCCESS: {script_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ ERROR: {script_path} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"✗ ERROR: {script_path} failed with exception: {e}")
        return False

def main():
    """Run all scripts in sequence."""
    failed = []
    
    for i, script in enumerate(SCRIPTS, 1):
        script_path = ROOT_DIR / script
        
        if not script_path.exists():
            print(f"✗ ERROR: Script not found: {script_path}")
            failed.append(script)
            continue
        
        print(f"\n[{i}/{len(SCRIPTS)}] {script}")
        
        success = run_script(script_path)
        
        if not success:
            failed.append(script)
            print(f"\n⚠ WARNING: Script {script} failed. Continuing with next script...")
    
    # Summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"Total scripts: {len(SCRIPTS)}")
    print(f"Successful: {len(SCRIPTS) - len(failed)}")
    print(f"Failed: {len(failed)}")
    
    if failed:
        print("\n⚠ Failed scripts:")
        for script in failed:
            print(f"  - {script}")
        print("\n❌ Pipeline completed with errors.")
        sys.exit(1)
    else:
        print("\n✓ All scripts executed successfully!")
        print("="*80)
        print("OUTPUT FILES:")
        print("="*80)
        print("\nCore Replication:")
        print("  Replication table (2007-2013): outputs/replication_table.txt")
        print("  Extension table (2007-2025): outputs/extension_table.txt")
        print("  Regression dataset: data/processed/regression_dataset.csv")
        print("\nExtension 1 - Learning Over Time:")
        print("  Learning regressions: outputs/extension1_learning_table.txt")
        print("  Learning LaTeX: outputs/extension1_learning_table.tex")
        print("\nExtension 2 - Risk vs Uncertainty:")
        print("  Risk/Uncertainty regressions: outputs/extension2_risk_uncertainty_table.txt")
        print("  Risk/Uncertainty LaTeX: outputs/extension2_risk_uncertainty_table.tex")
        print("\nCombined Extensions:")
        print("  Combined table: outputs/extensions_combined_table.txt")
        print("  Extended dataset: data/processed/regression_dataset_extended.csv")
        print("\nFigures:")
        print("  All figures: outputs/figures/")
        print("="*80)
        sys.exit(0)

if __name__ == "__main__":
    main()
