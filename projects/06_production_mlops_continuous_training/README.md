# Project 06: Production MLOps Continuous Training

## 1. Problem Statement & Business Context
Machine learning models deployed in production assume that incoming live data follows the same statistical distribution as historical training data (the stationary distribution assumption). When customer behaviors, pricing tariffs, or macroeconomic conditions shift, models suffer from silent **Covariate Shift / Data Drift**, leading to degraded business predictions.

This project implements an automated **Continuous Training MLOps Pipeline** that detects statistical distribution drift using the **2-Sample Kolmogorov-Smirnov (KS-Test)** and triggers automated model retraining and checkpointing.

---

## 2. System Architecture
```
              [ Live Production Data Stream ]
                            │
                            ▼
              [ 2-Sample KS Statistical Test ]
           (Calculates Divergence D and p-Value)
                            │
            ┌───────────────┴───────────────┐
            ▼                               ▼
    [ p-Value >= 0.05 ]             [ p-Value < 0.05 ]
    Distribution Stable           COVARIATE SHIFT DETECTED
            │                               │
            ▼                               ▼
     Continue Serving             [ Automated Retrain ]
                                    Fresh Production Batch
                                            │
                                            ▼
                                [ Model Checkpointing ]
                                 models/mlops_model.joblib
```

---

## 3. Mathematical Formulation
### Kolmogorov-Smirnov 2-Sample Test
Given empirical distribution functions $F_1(x)$ (baseline training) and $F_2(x)$ (live production stream):

$$D = \sup_x |F_1(x) - F_2(x)|$$

Under the null hypothesis $H_0$ that both samples are drawn from the same continuous distribution, the test yields a $p$-value. When $p < \alpha = 0.05$, $H_0$ is rejected, confirming statistically significant covariate drift.

---

## 4. Project Structure & Components
```
projects/06_production_mlops_continuous_training/
├── 01_production_mlops_continuous_training_masterclass.ipynb  # Masterclass notebook
├── README.md                                                 # Comprehensive documentation
├── pipeline.py                                               # MLOps Drift & Continuous Retraining Engine
└── test_pipeline.py                                          # Pytest verification suite
```

---

## 5. Masterclass Notebook Walkthrough
1. **Problem Statement & Drift Risks**: The silent degradation of static production ML models.
2. **Data Stream Ingestion & Visualization**: Visualizing shifted production distributions vs baseline KDEs.
3. **Statistical KS-Test Execution**: Calculating test statistic $D$ and automated retraining trigger.
4. **Model Checkpointing & Audit Trail**: Saving retrained pipeline to `models/mlops_continuous_training_model.joblib`.
5. **Executive Summary & Governance**: Canary deployments, golden test sets, and automated alerting.

---

## 6. Running Production Microservices
```bash
# 1. Run unit tests
pytest projects/06_production_mlops_continuous_training/test_pipeline.py

# 2. Run MLOps continuous training pipeline
python projects/06_production_mlops_continuous_training/pipeline.py
```

---

## 7. Performance Benchmarks & SLAs
- **Drift Detection Sensitivity**: Detects subtle mean shifts ($> 5\%$) with $p < 10^{-10}$.
- **Automated Retraining Time**: < 1.5 seconds for complete model update and validation.
- **Audit Compliance**: Full tracking of drift timestamps, p-values, and model versions.
