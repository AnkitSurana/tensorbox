"""Automated Test Suite for all Tensorbox Project and Journey Notebooks.

Executes all 10 Project masterclasses and all 10 Journey masterclasses
using nbconvert/nbclient to ensure 100% error-free execution.
"""

import os
import sys
from pathlib import Path
import nbformat
from nbclient import NotebookClient

tensorbox_root = Path(__file__).resolve().parent.parent

def test_notebook(nb_path: Path):
    print(f"\n▶ Testing: {nb_path.relative_to(tensorbox_root)}...")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    client = NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(nb_path.parent)}})
    client.execute()
    print(f"  ✓ PASSED: {nb_path.name}")

def main():
    print("=" * 70)
    print("🔍 TENSORBOX AUTOMATED NOTEBOOK TEST SUITE")
    print("=" * 70)
    
    # 1. Project Masterclasses
    project_nbs = sorted(list(tensorbox_root.glob("projects/**/*.ipynb")))
    print(f"\nFound {len(project_nbs)} Project masterclass notebooks.")
    
    # 2. Journey Masterclasses
    journey_nbs = sorted(list(tensorbox_root.glob("notebooks/**/*.ipynb")))
    print(f"Found {len(journey_nbs)} Journey Masterclass notebooks.")
    
    all_nbs = journey_nbs + project_nbs
    print(f"\nTotal notebooks to validate: {len(all_nbs)}")
    
    failed = []
    for nb in all_nbs:
        try:
            test_notebook(nb)
        except Exception as e:
            print(f"  ❌ FAILED: {nb.name}")
            print(f"     Error: {e}")
            failed.append((nb, str(e)))
            
    print("\n" + "=" * 70)
    if not failed:
        print(f"🎉 ALL {len(all_nbs)} NOTEBOOKS EXECUTED CLEANLY WITH 0 ERRORS!")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"⚠️ {len(failed)} NOTEBOOKS FAILED EXECUTION:")
        for nb, err in failed:
            print(f" - {nb.relative_to(tensorbox_root)}: {err[:100]}...")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()
