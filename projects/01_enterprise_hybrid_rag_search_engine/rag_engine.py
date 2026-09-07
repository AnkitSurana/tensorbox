"""
Hybrid RAG Engine implementation with BM25, Dense Cosine, and RRF Fusion.
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class HybridRAGEngine:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.tfidf = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = self.tfidf.fit_transform(documents)
        # Synthetic dense embeddings
        np.random.seed(42)
        self.dense_embeddings = np.random.randn(len(documents), 32)
        # Normalize
        self.dense_embeddings /= np.linalg.norm(self.dense_embeddings, axis=1, keepdims=True)

    def retrieve_sparse(self, query: str, top_k: int = 3) -> Dict[int, int]:
        q_vec = self.tfidf.transform([query])
        scores = (self.tfidf_matrix * q_vec.T).toarray().ravel()
        ranked = np.argsort(scores)[::-1][:top_k]
        return {doc_id: rank + 1 for rank, doc_id in enumerate(ranked)}

    def retrieve_dense(self, query: str, top_k: int = 3) -> Dict[int, int]:
        # Simulated dense representation of query
        q_vec = np.random.randn(1, 32)
        q_vec /= np.linalg.norm(q_vec)
        scores = cosine_similarity(q_vec, self.dense_embeddings)[0]
        ranked = np.argsort(scores)[::-1][:top_k]
        return {doc_id: rank + 1 for rank, doc_id in enumerate(ranked)}

    def hybrid_search(self, query: str, top_k: int = 3, rrf_k: int = 60) -> List[Tuple[int, float, str]]:
        sparse_ranks = self.retrieve_sparse(query, top_k=top_k*2)
        dense_ranks = self.retrieve_dense(query, top_k=top_k*2)
        
        all_doc_ids = set(sparse_ranks.keys()).union(set(dense_ranks.keys()))
        fusion_scores = {}
        for doc_id in all_doc_ids:
            score = 0.0
            if doc_id in sparse_ranks:
                score += 1.0 / (rrf_k + sparse_ranks[doc_id])
            if doc_id in dense_ranks:
                score += 1.0 / (rrf_k + dense_ranks[doc_id])
            fusion_scores[doc_id] = score
            
        ranked_results = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [(doc_id, score, self.documents[doc_id]) for doc_id, score in ranked_results]
