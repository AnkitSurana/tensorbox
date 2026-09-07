"""Tensorbox Kaggle Dataset Loader.

Provides seamless loading of Kaggle datasets from structured local folders or files.
Automatically resolves:
  1. Kaggle Folder Structure: data/<dataset>/train.csv, data/<dataset>/test.csv, data/<dataset>/<custom>.csv
  2. Direct CSV File: data/<dataset>.csv
  3. /workspace/data/<dataset>/
"""

import os
from pathlib import Path
from typing import Optional, Union
import pandas as pd
from utils.logger import get_logger

logger = get_logger("DataLoader")


def get_data_dir() -> Path:
    """Resolve the active data directory."""
    candidates = [
        Path("/workspace/data"),
        Path(__file__).parent.parent / "data",
        Path.cwd() / "data",
        Path.cwd().parent / "data"
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    default = Path("/workspace/data") if Path("/workspace").exists() and os.access("/workspace", os.W_OK) else Path.cwd() / "data"
    default.mkdir(parents=True, exist_ok=True)
    return default


def load_dataset(
    name: str,
    split: str = "train",
    filename: Optional[str] = None,
    **read_csv_kwargs
) -> Union[pd.DataFrame, Path]:
    """Load a dataset from the local data folder or direct file.

    Parameters
    ----------
    name : str
        The dataset or folder name (e.g. "titanic", "housing_prices", "bike_sharing").
    split : str
        The partition to load if organized as a Kaggle folder ("train", "test", "gender_submission", etc.). Default is "train".
    filename : Optional[str]
        Specific file to load within the dataset directory (e.g. "hour.csv").

    Returns
    -------
    pd.DataFrame or Path
    """
    data_dir = get_data_dir()
    clean_name = name.lower().replace("-", "_").replace(".csv", "").replace(".txt", "").replace(".jsonl", "")

    # Check text / document corpuses
    if clean_name == "knowledge_base":
        for candidate in [data_dir / "knowledge_base.txt", data_dir / "knowledge_base.md"]:
            if candidate.exists():
                return candidate
    if clean_name == "instruction_tuning":
        candidate = data_dir / "instruction_tuning.jsonl"
        if candidate.exists():
            return candidate

    # 1. Check Dataset Subfolder (Kaggle directory layout: data/<name>/train.csv)
    folder = data_dir / clean_name
    if folder.is_dir():
        # Priority A: exact requested filename
        if filename and (folder / filename).exists():
            target = folder / filename
            logger.info(f"Loading {clean_name}/{filename} from {target}")
            return pd.read_csv(target, **read_csv_kwargs)
        
        # Priority B: split name (train.csv, test.csv, etc.)
        split_file = folder / f"{split}.csv"
        if split_file.exists():
            logger.info(f"Loading {clean_name} [{split}] from {split_file}")
            return pd.read_csv(split_file, **read_csv_kwargs)
            
        # Priority C: any CSV inside folder
        csv_files = list(folder.glob("*.csv"))
        if csv_files:
            logger.info(f"Loading {clean_name} from {csv_files[0]}")
            return pd.read_csv(csv_files[0], **read_csv_kwargs)

    # 2. Check Flat CSV File: data/<clean_name>.csv
    flat_csv = data_dir / f"{clean_name}.csv"
    if flat_csv.exists():
        logger.info(f"Loading {clean_name} from {flat_csv}")
        return pd.read_csv(flat_csv, **read_csv_kwargs)

    # 3. Not found: raise actionable informative error
    err_msg = (
        f"\n❌ Kaggle Dataset '{clean_name}' was not found.\n\n"
        f"Expected location:\n"
        f"  📁 {data_dir}/{clean_name}/train.csv (or {data_dir}/{clean_name}.csv)\n\n"
        f"Instructions:\n"
        f"  1. Download the dataset from Kaggle.\n"
        f"  2. Place or unzip the files in '{data_dir}/{clean_name}/'.\n"
        f"  3. Re-run your code — it will be automatically discovered and loaded!\n"
    )
    raise FileNotFoundError(err_msg)


def list_available_datasets():
    """List all currently discovered datasets in the data folder."""
    data_dir = get_data_dir()
    print("\n======================================================================")
    print("📂 TENSORBOX LOCAL DATASET REPOSITORY")
    print("======================================================================")
    
    entries = sorted(list(data_dir.iterdir()))
    for item in entries:
        if item.name.startswith("."):
            continue
        if item.is_dir():
            files = [f.name for f in item.iterdir() if not f.name.startswith(".")]
            print(f" 📁 {item.name:<25} [Folder] Contains: {', '.join(files[:4])}")
        elif item.is_file():
            size_kb = item.stat().st_size / 1024
            print(f" 📄 {item.name:<25} [File]   {size_kb:.1f} KB")
    print("======================================================================\n")
