"""
Unit tests for Fraud Detector & Feature Store.
"""

import pytest
from feature_store import FeatureStore
from fraud_detector import FraudDetector

def test_feature_store_put_get():
    fs = FeatureStore()
    fs.put("acc_101", {"avg_spend_30d": 120.0, "failed_logins_24h": 0})
    assert fs.get("acc_101")["avg_spend_30d"] == 120.0
    assert fs.count() == 1

def test_fraud_scoring():
    fs = FeatureStore()
    fs.put("acc_999", {"avg_spend_30d": 20.0, "failed_logins_24h": 5})
    detector = FraudDetector(fs)
    res = detector.score_transaction("acc_999", amount=5000.0, location_diff=1.0)
    assert res["fraud_probability"] > 0.5
    assert res["decision"] in ["APPROVE", "DECLINE"]
