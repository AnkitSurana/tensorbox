"""
FastAPI Serving Endpoint for Hybrid RAG Search Engine.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from rag_engine import HybridRAGEngine

app = FastAPI(title="Enterprise Hybrid RAG API", version="1.0.0")

CORPUS = [
    "Tensorbox provides zero-setup Docker environments for AI and Machine Learning.",
    "PyTorch is an open source deep learning framework developed by Meta AI.",
    "FastAPI enables high-performance REST APIs with automatic OpenAPI Swagger documentation.",
    "BM25 is a ranking function used by search engines to estimate relevance of documents.",
    "Hybrid RAG combines dense vector retrieval with keyword BM25 indexing for superior accuracy.",
    "LoRA (Low-Rank Adaptation) reduces trainable parameters by decomposing weight update matrices."
]

engine = HybridRAGEngine(CORPUS)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 3

class SearchHit(BaseModel):
    document_id: int
    score: float
    content: str

class SearchResponse(BaseModel):
    query: str
    results: List[SearchHit]

@app.get("/health")
def health():
    return {"status": "healthy", "documents_indexed": len(CORPUS)}

@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    hits = engine.hybrid_search(req.query, top_k=req.top_k)
    return SearchResponse(
        query=req.query,
        results=[SearchHit(document_id=h[0], score=round(h[1], 5), content=h[2]) for h in hits]
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
