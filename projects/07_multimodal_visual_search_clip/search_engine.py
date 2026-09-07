"""
Multimodal CLIP Search Engine.
"""

import numpy as np
from typing import List, Dict, Any

class MultimodalSearchEngine:
    def __init__(self, dim: int = 32):
        self.dim = dim
        self.gallery: Dict[str, np.ndarray] = {}

    def index_image(self, image_id: str, embedding: np.ndarray):
        norm_emb = embedding / np.linalg.norm(embedding)
        self.gallery[image_id] = norm_emb

    def search_by_text(self, text_embedding: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
        norm_text = text_embedding / np.linalg.norm(text_embedding)
        results = []
        for img_id, img_emb in self.gallery.items():
            sim = float(np.dot(norm_text, img_emb))
            results.append({"image_id": img_id, "similarity": round(sim, 4)})
            
        return sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]
