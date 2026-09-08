# Tensorbox: Modular Machine Learning & AI Engineering Workstation

Tensorbox is an all-in-one, zero-setup machine learning and artificial intelligence development workstation. It provides a structured, reproducible curriculum of **10 Progressive Learning Journeys** and **10 Production-Grade Engineering Projects** built on local Kaggle-standard datasets with sub-millisecond execution benchmarks.

---

## 1. Architectural Overview & Workstation Capabilities

```
================================================================================
                                TENSORBOX WORKSTATION
================================================================================
  [ Local Dataset Layer ]      --> 12 Kaggle-Standard Datasets (data/*)
  [ Learning Journeys ]        --> 10 Step-by-Step Educational Notebooks (notebooks/*)
  [ Production Projects ]      --> 10 Standalone Microservice Projects (projects/*)
  [ Model Artifact Registry ]  --> Persistent Serialized Estimators (models/*)
  [ Enterprise CLI & Utils ]   --> Dataset Ingestion, Vector DBs, Health Checks (utils/*)
  [ Verification Suite ]       --> Full PyTest & Notebook Execution Engines (tests/*)
================================================================================
```

### Core Interfaces & Services
- **JupyterLab Development**: Interactive exploratory data analysis, algorithm derivation, and model training.
- **FastAPI / OpenAPI Microservices**: Production REST inference endpoints with automatic Swagger documentation.
- **In-Memory Feature Stores & Vector DBs**: Sub-millisecond similarity retrieval (Qdrant / ChromaDB / TF-IDF).
- **Persistent Model Artifact Registry**: Automated serialization of champion models into `models/` (.joblib and .pt).

---

## 2. Masterclass Learning Journeys (notebooks/)

The `notebooks/` directory contains 10 foundational, self-contained educational journeys designed with kid-friendly real-world analogies, step-by-step mathematical derivations, output breakdowns, and executive summaries:

| Journey | Module Title | Core Machine Learning Concepts | Artifact Output |
| :--- | :--- | :--- | :--- |
| **Journey 00** | Environment Setup & Ingestion | Environment validation, Kaggle schemas, Memory profiling | System certified |
| **Journey 01** | Titanic Survival Classification | Gini Impurity, Decision Trees, Tree Depth Sweeps | `models/titanic_best_model.joblib` |
| **Journey 02** | Housing Prices Regression | Ordinary Least Squares, Ridge (L2) vs Lasso (L1) | `models/housing_best_model.joblib` |
| **Journey 03** | Telecom Churn Imbalance | Precision/Recall, Asymmetric Cost Functions | `models/telecom_churn_best_model.joblib` |
| **Journey 04** | Bike Sharing Demand Forecasting | Cyclical Trigonometric Time (sin/cos), Ensembles | `models/bike_sharing_best_model.joblib` |
| **Journey 05** | Credit Fraud Anomaly Detection | Isolation Forest Tree Path Lengths, PR-AUC | `models/credit_fraud_best_model.joblib` |
| **Journey 06** | Customer Segmentation Clustering | Euclidean Distance, K-Means Elbow & Silhouette | `models/customer_segmentation_best_model.joblib` |
| **Journey 07** | Stock Market Trading RL | Markov Decision Processes, Q-Learning, Bellman Math | `models/stock_market_best_model.joblib` |
| **Journey 08** | NLP News & Sentiment | Tokenization, TF-IDF Vector Spaces, Cosine Similarity | `models/nlp_sentiment_best_model.joblib` |
| **Journey 09** | Ecommerce Recommender Systems | Truncated SVD, PyTorch Neural Two-Tower Embeddings | `models/ecommerce_recommender_best_model.pt` |
| **Journey 10** | GenAI, RAG & Tool Calling Agents | Semantic Chunking, Dynamic JSON Function Dispatch | `models/genai_agents_best_model.joblib` |

---

## 3. Production Engineering Projects (projects/)

The `projects/` directory contains 10 complete enterprise projects. Each project contains an end-to-end Masterclass Notebook (`01_*_masterclass.ipynb`), standalone Python microservices (`app.py`, `service.py`), and dedicated PyTest verification suites (`test_*.py`):

| Project | System Architecture | Mathematical & Engineering Focus | Production Files |
| :--- | :--- | :--- | :--- |
| **Project 01** | Enterprise Hybrid RAG Search Engine | BM25 Lexical + Dense Latent SVD + Reciprocal Rank Fusion ($k=60$) | `app.py`, `rag_engine.py`, `test_rag.py` |
| **Project 02** | Realtime Fraud Detection & Feature Store | Streaming Amount Z-Score Velocity + Isolation Forest Anomaly Scoring | `feature_store.py`, `fraud_detector.py`, `test_fraud.py` |
| **Project 03** | Autonomous Multi-Agent Market Analyst | Specialized Swarm (Technical, Valuation, Risk) with Deterministic Tools | `swarm.py`, `test_swarm.py` |
| **Project 04** | Two-Tower Recommendation Engine | PyTorch Dual Embedding Towers (16-dim) with Sub-Millisecond Top-K MIPS | `model.py`, `test_two_tower.py` |
| **Project 05** | Medical Image Segmentation & Grad-CAM | MRI Pathology Segmentation, Sørensen-Dice Metric, Grad-CAM Overlays | `unet.py`, `test_unet.py` |
| **Project 06** | Continuous Training MLOps Pipeline | 2-Sample Kolmogorov-Smirnov Covariate Drift Detection & Retrain Loops | `pipeline.py`, `test_pipeline.py` |
| **Project 07** | Multimodal Visual Search (CLIP) | Cross-Modal Normalized 32-dim Cosine Retrieval for Text-to-Image | `search_engine.py`, `test_multimodal.py` |
| **Project 08** | LLM Fine-Tuning LoRA Serving Hub | Low-Rank Decomposition ($W_0 + BA$) with 99.2% VRAM Memory Savings | `lora_server.py`, `test_lora_hub.py` |
| **Project 09** | Algorithmic Trading Reinforcement Learning | 3-State Regime MDP Q-Learning with Stop-Loss Risk Guardrails | `trading_env.py`, `test_trading.py` |
| **Project 10** | Voice & Document AI Assistant | Spoken Intent Parsing + TF-IDF Semantic Passage Search Engine | `assistant.py`, `test_assistant.py` |

---

## 4. Local Dataset Catalog (data/)

All datasets reside locally in clean, Kaggle-standard directories under `data/`:

| Dataset Name | Domain | Files Included | Target / Primary Features |
| :--- | :--- | :--- | :--- |
| `titanic` | Passenger Demographics | `train.csv`, `test.csv` | `Survived` (Binary Classification) |
| `housing_prices` | Residential Real Estate | `train.csv`, `test.csv` | `medv` (Median Property Value in $1k) |
| `telecom_churn` | Subscription Billing | `train.csv` | `Churn` (Imbalanced Binary Target) |
| `bike_sharing` | Urban Mobility Telemetry | `train.csv`, `test.csv` | `cnt` (Hourly Fleet Rental Demand) |
| `credit_fraud` | Card Transactions | `train.csv` | `Class` (0.17% Rare Anomaly Flag) |
| `customer_segmentation` | Mall Shopper Registry | `train.csv` | `Annual Income`, `Spending Score` |
| `stock_market` | Daily Equity History | `train.csv` | `AAPL.Close`, Moving Averages |
| `sentiment_dataset` | Customer Reviews | `train.csv` | `sentiment` (Positive vs Negative) |
| `movie_ratings` | User-Item Interactions | `train.csv` | `rating` (1.0 to 5.0 Star Ratings) |
| `news_articles` | Topic Documentation | `train.csv` | `text` (Unstructured News Corpus) |
| `knowledge_base` | Technical Documentation | `data.txt` | Unstructured Enterprise Manuals |
| `instruction_tuning` | LLM Fine-Tuning | `data.jsonl` | Instruction-Response JSONL Pairs |

---

## 5. Repository Directory Structure

```
tensorbox/
├── data/                                 # 12 Kaggle-standard local datasets
├── docs/                                 # Architecture diagrams and system specs
├── models/                               # Serialized production model artifacts
├── notebooks/                            # 10 Step-by-step masterclass journeys
│   ├── 00_setup_and_data_ingestion.ipynb
│   ├── 01_titanic_survival_journey/
│   ├── 02_housing_prices_journey/
│   ├── 03_telecom_churn_journey/
│   ├── 04_bike_sharing_journey/
│   ├── 05_credit_fraud_journey/
│   ├── 06_customer_segmentation_journey/
│   ├── 07_stock_market_trading_journey/
│   ├── 08_nlp_news_and_sentiment_journey/
│   ├── 09_ecommerce_recommender_journey/
│   └── 10_genai_rag_and_agents_journey/
├── projects/                             # 10 Production engineering projects
│   ├── 01_enterprise_hybrid_rag_search_engine/
│   ├── 02_realtime_fraud_detection_feature_store/
│   ├── 03_autonomous_multi_agent_market_analyst/
│   ├── 04_two_tower_ecommerce_recommendation_engine/
│   ├── 05_medical_image_segmentation_gradcam/
│   ├── 06_production_mlops_continuous_training/
│   ├── 07_multimodal_visual_search_clip/
│   ├── 08_llm_fine_tuning_lora_serving_hub/
│   ├── 09_algorithmic_trading_reinforcement_learning/
│   └── 10_voice_and_document_ai_assistant/
├── scripts/                              # Automated testing and dependency checkers
│   ├── check_dependencies.py
│   └── test_all_notebooks.py
├── tests/                                # Unit test suite for CLI and vector DBs
├── utils/                                # Data loaders, loggers, vector DB helpers
├── Dockerfile                            # Production container definition
├── docker-compose.yml                    # Multi-container orchestration
├── requirements.in                       # Top-level dependency specifications
├── requirements.txt                      # Pinned production lockfile
└── setup.py                              # Pip installable package setup
```

---

## 6. Quickstart & Installation

### Option A: Local Python Environment
```bash
# 1. Clone repository
git clone https://github.com/ankitsurana/tensorbox.git
cd tensorbox

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify system environment
python scripts/check_dependencies.py
```

### Option B: Docker Container
```bash
# Build and launch complete workstation
docker-compose up -d --build

# Access JupyterLab at http://localhost:8888
# Access FastAPI at http://localhost:5000/docs
```

---

## 7. Testing & Verification Suite

Tensorbox includes a comprehensive automated testing suite:

```bash
# Run unit test suite
pytest tests/

# Execute all 21 masterclass notebooks end-to-end
python scripts/test_all_notebooks.py
```

---

## 8. License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.
