"""
Automated unit tests for Hybrid RAG Engine.
"""

import pytest
from rag_engine import HybridRAGEngine

@pytest.fixture
def sample_engine():
    corpus = [
        "Machine learning models need quality training data.",
        "Deep learning leverages multi-layer neural networks.",
        "Docker containers ensure reproducible deployment environments."
    ]
    return HybridRAGEngine(corpus)

def test_engine_initialization(sample_engine):
    assert len(sample_engine.documents) == 3

def test_hybrid_search(sample_engine):
    results = sample_engine.hybrid_search("machine learning", top_k=2)
    assert len(results) <= 2
    assert results[0][1] > 0.0  # RRF score positive
    assert isinstance(results[0][2], str)
