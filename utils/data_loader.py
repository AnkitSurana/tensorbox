"""Tensorbox Kaggle Dataset Loader.

Provides seamless loading of Kaggle datasets from structured local folders.
Automatically resolves:
  1. Kaggle Folder Structure: data/<dataset>/train.csv, data/<dataset>/test.csv, data/<dataset>/data.csv
  2. Text / JSONL Corpuses: data/<dataset>/data.txt, data/<dataset>/data.jsonl
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
    """Load a dataset from the local data folder.

    Parameters
    ----------
    name : str
        The dataset folder name (e.g. "titanic", "housing_prices", "telecom_churn").
    split : str
        The partition to load if organized as a folder ("train", "test", etc.). Default is "train".
    filename : Optional[str]
        Specific file to load within the dataset directory (e.g. "data.txt", "train.csv").

    Returns
    -------
    pd.DataFrame or Path
    """
    data_dir = get_data_dir()
    clean_name = name.lower().replace("-", "_").replace(".csv", "").replace(".txt", "").replace(".jsonl", "")

    folder = data_dir / clean_name

    # Check if folder exists
    if folder.is_dir():
        # Text/JSONL corpuses
        if clean_name in ["knowledge_base", "instruction_tuning"]:
            for f in folder.iterdir():
                if f.is_file() and not f.name.startswith("."):
                    return f

        # Priority 1: exact requested filename
        if filename and (folder / filename).exists():
            target = folder / filename
            logger.info(f"Loading {clean_name}/{filename} from {target}")
            return pd.read_csv(target, **read_csv_kwargs)
        
        # Priority 2: split name (train.csv, test.csv, etc.)
        split_file = folder / f"{split}.csv"
        if split_file.exists():
            logger.info(f"Loading {clean_name} [{split}] from {split_file}")
            return pd.read_csv(split_file, **read_csv_kwargs)
            
        # Priority 3: any CSV inside folder
        csv_files = list(folder.glob("*.csv"))
        if csv_files:
            logger.info(f"Loading {clean_name} from {csv_files[0]}")
            return pd.read_csv(csv_files[0], **read_csv_kwargs)

        # Priority 4: any text or jsonl file inside folder
        all_files = [f for f in folder.iterdir() if f.is_file() and not f.name.startswith(".")]
        if all_files:
            return all_files[0]

    # Check flat file fallback if exists
    flat_csv = data_dir / f"{clean_name}.csv"
    if flat_csv.exists():
        logger.info(f"Loading {clean_name} from {flat_csv}")
        return pd.read_csv(flat_csv, **read_csv_kwargs)

    err_msg = (
        f"\n❌ Kaggle Dataset Folder '{clean_name}' was not found in '{data_dir}'.\n\n"
        f"Expected folder layout:\n"
        f"  📁 {data_dir}/{clean_name}/train.csv\n\n"
        f"Instructions:\n"
        f"  1. Download the dataset from Kaggle.\n"
        f"  2. Place the CSV files in '{data_dir}/{clean_name}/'.\n"
        f"  3. Re-run your code — it will be automatically discovered and loaded!\n"
    )
    raise FileNotFoundError(err_msg)


def list_available_datasets():
    """List all currently discovered datasets in the data folder."""
    data_dir = get_data_dir()
    print("\n======================================================================")
    print("📂 TENSORBOX LOCAL DATASET REPOSITORY (KAGGLE FOLDERS)")
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
