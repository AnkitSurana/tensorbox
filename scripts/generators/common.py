"""
Common utilities and helpers for curriculum notebook generators.
"""

from pathlib import Path
import nbformat as nbf

WORKSPACE_ROOT = Path("/Users/ankitsurana/Documents/tensorbox")
NOTEBOOKS_DIR = WORKSPACE_ROOT / "notebooks"
PROJECTS_DIR = WORKSPACE_ROOT / "projects"

def create_notebook(title: str, description: str, track_num: str, track_name: str, cells_data: list):
    """
    Creates a Jupyter notebook with standard metadata and formatting.
    """
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
    
    # Header cell
    header_md = f"""# {title}

**Track {track_num}: {track_name}** | *Tensorbox AI/ML Production Curriculum*

---
### Overview & Objectives
{description}
"""
    nb.cells.append(nbf.v4.new_markdown_cell(header_md))
    
    for cell_type, content in cells_data:
        if cell_type == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(content))
        elif cell_type == "code":
            nb.cells.append(nbf.v4.new_code_cell(content.strip()))
            
    return nb

def save_notebook(nb, rel_path: str):
    full_path = NOTEBOOKS_DIR / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"✓ Created notebook: notebooks/{rel_path}")

DATA_LOADER_BOILERPLATE = """
import os
import sys
from pathlib import Path

for p in [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
    if (p / "utils").exists():
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
        break

from utils.data_loader import load_dataset
"""

def make_file(rel_path: str, content: str):
    full_path = WORKSPACE_ROOT / rel_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    with open(full_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    print(f"✓ Created file: {rel_path}")
