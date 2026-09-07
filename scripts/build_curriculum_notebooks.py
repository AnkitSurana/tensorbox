"""
Curriculum Notebook Builder for Tensorbox.
Generates all 37 comprehensive Jupyter notebooks across 14 tracks.
"""

import os
import nbformat as nbf
from pathlib import Path

WORKSPACE_ROOT = Path("/Users/ankitsurana/Documents/student-ml-env")
NOTEBOOKS_DIR = WORKSPACE_ROOT / "notebooks"

def make_nb(title, description, cells_data):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    }
    
    # Title cell
    header_md = f"# {title}\n\n{description}\n\n---\n*Tensorbox AI/ML Production Curriculum*"
    nb.cells.append(nbf.v4.new_markdown_cell(header_md))
    
    for cell_type, content in cells_data:
        if cell_type == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "code":
            nb.cells.append(nbf.v4.new_code_cell(content.strip()))
            
    return nb

def save_nb(nb, rel_path):
    full_path = NOTEBOOKS_DIR / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated: {rel_path}")

print("Notebook builder framework initialized.")
