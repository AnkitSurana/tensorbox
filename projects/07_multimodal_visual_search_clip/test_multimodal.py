"""
Tests for Multimodal CLIP Search.
"""

import numpy as np
from search_engine import MultimodalSearchEngine

def test_multimodal_search():
    engine = MultimodalSearchEngine(dim=16)
    np.random.seed(42)
    engine.index_image("car.jpg", np.random.randn(16))
    engine.index_image("dog.jpg", np.random.randn(16))
    
    hits = engine.search_by_text(np.random.randn(16), top_k=2)
    assert len(hits) == 2
    assert "similarity" in hits[0]
