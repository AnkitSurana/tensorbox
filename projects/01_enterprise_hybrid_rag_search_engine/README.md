# Enterprise Hybrid RAG Search Engine

Production-grade Hybrid Retrieval-Augmented Generation system combining **BM25 Lexical Keyword Search**, **Dense Vector Semantic Embeddings**, **Reciprocal Rank Fusion (RRF)**, and **Cross-Encoder Reranking**.

## Architecture
```
[User Query]
     │
     ├──► [BM25 Inverted Index] ──► Top-K Lexical Candidates ─┐
     │                                                        ├─► [RRF Fusion] ──► [Cross-Encoder] ──► [FastAPI / LLM]
     └──► [Dense Vector Index] ───► Top-K Semantic Candidates ┘
```

## Quickstart
```bash
# Run automated tests
pytest tests/

# Run FastAPI backend
python app.py
```
