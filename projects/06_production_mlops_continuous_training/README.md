# Production MLOps Continuous Training Pipeline

Automated ML orchestration pipeline:
- **Data Validation & Ingestion**: Schema verification and null checks.
- **Model Training & Evaluation**: Train candidate model vs production baseline.
- **Champion-Challenger Promotion Gate**: Model promotes only if validation F1 exceeds production threshold.
- **Prometheus Metric Exporter**: Emits latency, accuracy, and throughput metrics.
