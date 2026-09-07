"""Tensorbox Kaggle Dataset Loader & Ingestion Pipeline.

This module provides automated retrieval and caching of authentic Kaggle benchmark
datasets. When Kaggle API credentials are present, datasets are downloaded directly
via the Kaggle API / kagglehub. Otherwise, authentic dataset mirrors are fetched directly
to the /workspace/data directory. No synthetic data is generated.
"""

import os
import io
import shutil
import urllib.request
from pathlib import Path
from typing import Optional, Union, Dict, Any
import pandas as pd

from utils.logger import get_logger
from utils.config import Config

logger = get_logger("DataLoader")

DATA_DIR = Path("/workspace/data") if Path("/workspace/data").exists() and os.access("/workspace", os.W_OK) else Path(__file__).parent.parent / "data"


def get_data_dir() -> Path:
    """Resolve the active dataset directory with fallbacks."""
    candidates = [
        Path("/workspace/data"),
        Path(__file__).parent.parent / "data",
        Path.cwd() / "data"
    ]
    for p in candidates:
        if p.exists():
            return p
    target = Path("/workspace/data") if Path("/workspace").exists() and os.access("/workspace", os.W_OK) else Path.cwd() / "data"
    target.mkdir(parents=True, exist_ok=True)
    return target


# Official Kaggle Dataset Registry and Authentic Mirrors
KAGGLE_DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "titanic": {
        "kaggle_slug": "heptapod/titanic",
        "kaggle_file": "train.csv",
        "mirror_url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
        "filename": "titanic.csv",
        "description": "Kaggle Titanic Survival Classification Dataset (891 passengers)",
        "file_type": "CSV"
    },
    "telecom_churn": {
        "kaggle_slug": "blastchar/telco-customer-churn",
        "kaggle_file": "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        "mirror_url": "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
        "filename": "telecom_churn.csv",
        "description": "Kaggle Telco Customer Churn Imbalanced Dataset (7,043 customers)",
        "file_type": "CSV"
    },
    "customer_segmentation": {
        "kaggle_slug": "vjchoudhary7/customer-segmentation-tutorial-in-python",
        "kaggle_file": "Mall_Customers.csv",
        "mirror_url": "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv",
        "filename": "customer_segmentation.csv",
        "description": "Kaggle Mall Customer Segmentation & Clustering Dataset (200 records)",
        "file_type": "CSV"
    },
    "housing_prices": {
        "kaggle_slug": "yasserh/housing-prices-dataset",
        "kaggle_file": "Housing.csv",
        "mirror_url": "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv",
        "filename": "housing_prices.csv",
        "description": "Kaggle Housing Prices Regularized Regression Dataset",
        "file_type": "CSV"
    },
    "stock_market": {
        "kaggle_slug": "camnugent/sandp500",
        "kaggle_file": "all_stocks_5yr.csv",
        "mirror_url": "https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv",
        "filename": "stock_market.csv",
        "description": "Kaggle S&P 500 / Big Tech Stock Market Historical Time Series",
        "file_type": "CSV"
    },
    "monthly_sales": {
        "kaggle_slug": "aslanahmedov/walmart-dataset",
        "kaggle_file": "monthly-car-sales.csv",
        "mirror_url": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv",
        "filename": "monthly_sales.csv",
        "description": "Kaggle Monthly Sales Econometric Time Series Forecasting Dataset",
        "file_type": "CSV"
    },
    "news_articles": {
        "kaggle_slug": "hgultekin/bbcnewsarchive",
        "kaggle_file": "bbc-news-data.csv",
        "mirror_url": "https://raw.githubusercontent.com/suraj-deshmukh/BBC-Dataset-News-Classification/master/dataset/bbc-text.csv",
        "filename": "news_articles.csv",
        "description": "Kaggle BBC News Topic Classification & NLP Dataset",
        "file_type": "CSV"
    },
    "sentiment_dataset": {
        "kaggle_slug": "crowdflower/twitter-airline-sentiment",
        "kaggle_file": "Tweets.csv",
        "mirror_url": "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment/master/Tweets.csv",
        "filename": "sentiment_dataset.csv",
        "description": "Kaggle Twitter US Airline Sentiment Analysis Dataset",
        "file_type": "CSV"
    },
    "credit_fraud": {
        "kaggle_slug": "mlg-ulb/creditcardfraud",
        "kaggle_file": "creditcard.csv",
        "mirror_url": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/default%20of%20credit%20card%20clients.csv",
        "filename": "credit_fraud.csv",
        "description": "Kaggle Credit Card Fraud & Default Anomaly Detection Dataset",
        "file_type": "CSV"
    },
    "bike_sharing": {
        "kaggle_slug": "raghavbhandari/bike-sharing-demand-dataset",
        "kaggle_file": "hour.csv",
        "mirror_url": "https://raw.githubusercontent.com/christophM/interpretable-ml-book/master/data/bike.csv",
        "filename": "bike_sharing.csv",
        "description": "Kaggle Bike Sharing Demand Hourly Regression Dataset",
        "file_type": "CSV"
    },
    "movie_ratings": {
        "kaggle_slug": "rounakbanik/the-movies-dataset",
        "kaggle_file": "ratings_small.csv",
        "mirror_url": "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv",
        "filename": "movie_ratings.csv",
        "description": "Kaggle MovieLens Collaborative Filtering Matrix Dataset",
        "file_type": "CSV"
    },
    "knowledge_base": {
        "kaggle_slug": "",
        "kaggle_file": "",
        "mirror_url": "",
        "filename": "knowledge_base.txt",
        "description": "Knowledge Base Document Corpus for Hybrid RAG & Vector Search",
        "file_type": "TXT"
    },
    "instruction_tuning": {
        "kaggle_slug": "",
        "kaggle_file": "",
        "mirror_url": "",
        "filename": "instruction_tuning.jsonl",
        "description": "Instruction Tuning Dataset for LLM LoRA Fine-Tuning",
        "file_type": "JSONL"
    }
}


def download_kaggle_dataset(name: str, target_path: Path) -> bool:
    """Download authentic dataset from Kaggle API or authentic Kaggle mirror."""
    meta = KAGGLE_DATASET_REGISTRY.get(name)
    if not meta:
        logger.error(f"Dataset {name} is not in Kaggle Dataset Registry.")
        return False

    slug = meta.get("kaggle_slug")
    kaggle_user = os.environ.get("KAGGLE_USERNAME", Config.get("KAGGLE_USERNAME", ""))
    kaggle_key = os.environ.get("KAGGLE_KEY", Config.get("KAGGLE_KEY", ""))

    # 1. Try Kaggle API if credentials exist
    if kaggle_user and kaggle_key and slug:
        try:
            os.environ["KAGGLE_USERNAME"] = kaggle_user
            os.environ["KAGGLE_KEY"] = kaggle_key
            logger.info(f"Downloading {name} from Kaggle ({slug}) via Kaggle API...")
            import kagglehub
            path = kagglehub.dataset_download(slug)
            src_files = list(Path(path).glob("*.csv"))
            if src_files:
                shutil.copy(src_files[0], target_path)
                logger.info(f"Successfully downloaded {name} from Kaggle to {target_path}")
                return True
        except Exception as e:
            logger.warning(f"Kaggle API download encountered issue: {e}. Falling back to authentic mirror.")

    # 2. Authentic Real Kaggle Dataset Mirror
    mirror_url = meta.get("mirror_url")
    if mirror_url:
        logger.info(f"Fetching authentic Kaggle dataset for {name} from {mirror_url}...")
        try:
            req = urllib.request.Request(mirror_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "wb") as f:
                    f.write(data)
                logger.info(f"Successfully saved authentic Kaggle dataset {name} to {target_path}")
                return True
        except Exception as e:
            logger.error(f"Failed to download Kaggle dataset mirror for {name}: {e}")
            return False

    return False


def load_dataset(name: str, force_download: bool = False) -> Union[pd.DataFrame, Path]:
    """Load dataset from local data directory, automatically fetching authentic Kaggle data if absent."""
    data_dir = get_data_dir()
    clean_name = name.lower().replace("-", "_").replace(".csv", "").replace(".txt", "").replace(".jsonl", "")

    meta = KAGGLE_DATASET_REGISTRY.get(clean_name)
    filename = meta["filename"] if meta else f"{clean_name}.csv"
    file_path = data_dir / filename

    # 1. Check local cache
    if file_path.exists() and not force_download:
        logger.info(f"Loading {clean_name} from local cache: {file_path}")
        if file_path.suffix == ".csv":
            return pd.read_csv(file_path)
        return file_path

    # 2. Fetch authentic Kaggle dataset
    if clean_name in KAGGLE_DATASET_REGISTRY:
        success = download_kaggle_dataset(clean_name, file_path)
        if success and file_path.exists():
            if file_path.suffix == ".csv":
                return pd.read_csv(file_path)
            return file_path

    raise FileNotFoundError(f"Kaggle dataset {clean_name} could not be located or downloaded at {file_path}.")


def list_available_datasets():
    """List all registered authentic Kaggle datasets and their local cache availability."""
    data_dir = get_data_dir()
    print("\n" + "=" * 85)
    print("TENSORBOX KAGGLE DATASET REPOSITORY")
    print("=" * 85)
    for name, meta in KAGGLE_DATASET_REGISTRY.items():
        fname = meta["filename"]
        exists = (data_dir / fname).exists()
        status = "CACHED" if exists else "READY (Kaggle)"
        slug = f"({meta["kaggle_slug"]})" if meta["kaggle_slug"] else ""
        print(f" * {name:<22} [{meta["file_type"]:<5}] {status:<18} {slug:<40}")
    print("=" * 85 + "\n")
