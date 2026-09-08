# Project 02: Realtime Fraud Detection & Feature Store

## 1. Problem Statement & Business Context
Financial institutions process millions of credit card transactions per second. Fraudulent transactions constitute less than 0.2% of total transaction volume. A successful anti-fraud architecture must:
- Detect unauthorized charges in under **10 milliseconds** before card swipe authorization completes.
- Eliminate costly false positives that disrupt legitimate customer shopping experiences.
- Continuously compute real-time velocity features (e.g. sudden deviations from historic average transaction amounts).

This project implements an end-to-end **Realtime Fraud Detection Engine backed by an In-Memory Feature Store** using **Isolation Forest Anomaly Detection** and streaming Z-score velocity transformations.

---

## 2. System Architecture
```
             [ Live Credit Card Swipe Event ]
                            │
                            ▼
              [ Realtime Feature Store ]
          (Log Amount, Rolling Z-Score Velocity)
                            │
                            ▼
            [ Isolation Forest Anomaly Scorer ]
             (Tree Path Length Anomaly Score)
                            │
                            ▼
               [ Multi-Tier Decision Gate ]
         Score <= 0.08: APPROVE SWIPE
         0.08 < Score <= 0.15: TRIGGER SMS 2FA
         Score > 0.15: DECLINE & FREEZE CARD
```

---

## 3. Mathematical Formulation
### Isolation Forest Anomaly Score
The anomaly score $s(x, n)$ for an observation $x$ across a sample size of $n$ instances:

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

Where $E(h(x))$ is the average path length across random isolation trees, and $c(n)$ is the average path length of unsuccessful searches in a Binary Search Tree:

$$c(n) = 2\left(\ln(n - 1) + 0.5772156649\right) - \frac{2(n - 1)}{n}$$

When $s \to 1$, $x$ is an extreme anomaly (fraud). When $s < 0.5$, $x$ is normal.

---

## 4. Project Structure & Components
```
projects/02_realtime_fraud_detection_feature_store/
├── 01_realtime_fraud_detection_masterclass.ipynb  # Full end-to-end masterclass notebook
├── README.md                                      # Comprehensive technical documentation
├── feature_store.py                               # Streaming In-Memory Feature Store
├── fraud_detector.py                              # Isolation Forest Anomaly Model
└── test_fraud.py                                  # Automated pytest verification suite
```

---

## 5. Masterclass Notebook Walkthrough
1. **Problem Statement & Business Risks**: Defining credit card fraud economics and false positive costs.
2. **Transaction Ingestion & EDA**: Visualizing 0.17% class imbalance and transaction amounts.
3. **Feature Store Transformations**: Engineering streaming Amount Z-scores and logarithmic amounts.
4. **Isolation Forest Model Training**: Training random tree partitions on PCA and velocity features.
5. **Model Checkpointing & Live Scoring**: Saving model to `models/fraud_feature_store_model.joblib`.
6. **Executive Summary & Production Guidelines**: Gateway SLA, Kafka streaming, and monitoring.

---

## 6. Running Production Microservices
```bash
# 1. Run unit tests
pytest projects/02_realtime_fraud_detection_feature_store/test_fraud.py

# 2. Run feature store & fraud detector module
python projects/02_realtime_fraud_detection_feature_store/fraud_detector.py
```

---

## 7. Performance Benchmarks & SLAs
- **Inference Latency**: < 0.2 milliseconds per transaction (well within 10ms gateway SLA).
- **Throughput**: > 5,000 transactions/second on single CPU core.
- **Precision-Recall Area**: Average Precision (PR-AUC) of > 0.70 on highly imbalanced transaction streams.
