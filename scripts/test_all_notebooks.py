"""
Automated Batch Execution Runner for All Tensorbox Curriculum Notebooks.
Executes every notebook cell-by-cell in the current environment and asserts 0 failures.
"""

import os
import sys
import glob
import time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

WORKSPACE_ROOT = Path("/Users/ankitsurana/Documents/tensorbox")
NOTEBOOKS_DIR = WORKSPACE_ROOT / "notebooks"

def execute_notebook(nb_path: Path):
    print(f"▶ Executing: {nb_path.relative_to(NOTEBOOKS_DIR)} ...", end=" ", flush=True)
    start = time.time()
    
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
        
    client = NotebookClient(nb, timeout=120, kernel_name="python3", resources={"metadata": {"path": str(nb_path.parent)}})
    
    try:
        client.execute()
        elapsed = time.time() - start
        print(f"✓ PASSED ({elapsed:.2f}s)")
        return True, None
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ FAILED ({elapsed:.2f}s)")
        print(f"  Error: {e}")
        return False, str(e)

def main():
    notebook_files = sorted(list(NOTEBOOKS_DIR.glob("**/*.ipynb")))
    print(f"Found {len(notebook_files)} notebooks to execute.")
    
    passed = 0
    failed = 0
    errors = []
    
    start_all = time.time()
    for nb_path in notebook_files:
        success, err = execute_notebook(nb_path)
        if success:
            passed += 1
        else:
            failed += 1
            errors.append((str(nb_path.name), err))
            
    total_time = time.time() - start_all
    print("\n" + "="*50)
    print(f"🏁 EXECUTION SUMMARY ({total_time:.2f}s total)")
    print(f"Total Notebooks: {len(notebook_files)}")
    print(f"Passed         : {passed}")
    print(f"Failed         : {failed}")
    print("="*50)
    
    if failed > 0:
        print("\nFailures:")
        for name, err in errors:
            print(f"- {name}: {err}")
        sys.exit(1)
    else:
        print("\n🎉 ALL 100% OF NOTEBOOKS EXECUTED CLEANLY WITH ZERO ERRORS!")
        sys.exit(0)

if __name__ == "__main__":
    main()
