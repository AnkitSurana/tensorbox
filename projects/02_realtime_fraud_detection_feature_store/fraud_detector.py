"""
Fraud Scoring Engine with Explainability.
"""

import numpy as np
from typing import Dict, Any, Tuple
from feature_store import FeatureStore

class FraudDetector:
    def __init__(self, feature_store: FeatureStore):
        self.fs = feature_store

    def score_transaction(self, account_id: str, amount: float, location_diff: float) -> Dict[str, Any]:
        user_feats = self.fs.get(account_id) or {
            "avg_spend_30d": 50.0,
            "failed_logins_24h": 0,
            "risk_tier": 1
        }
        
        # Scoring logic
        spend_ratio = amount / (user_feats["avg_spend_30d"] + 1.0)
        raw_risk = (spend_ratio * 0.4) + (user_feats["failed_logins_24h"] * 0.3) + (location_diff * 0.3)
        fraud_prob = 1.0 / (1.0 + np.exp(-raw_risk + 2.0))
        
        is_fraud = fraud_prob >= 0.70
        return {
            "account_id": account_id,
            "fraud_probability": round(float(fraud_prob), 4),
            "is_fraud": bool(is_fraud),
            "decision": "DECLINE" if is_fraud else "APPROVE",
            "top_factors": {
                "spend_ratio": round(float(spend_ratio), 2),
                "failed_logins": user_feats["failed_logins_24h"]
            }
        }
