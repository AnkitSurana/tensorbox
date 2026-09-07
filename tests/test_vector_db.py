"""Tests for Vector Database Manager and Configs."""
from unittest.mock import patch, MagicMock
from utils.vector_db import VectorDBConfig, VectorDBManager


def test_vector_db_config():
    """Verify vector DB configuration values."""
    assert VectorDBConfig.CHROMA_PORT == 8000
    assert VectorDBConfig.QDRANT_PORT == 6333
    assert VectorDBConfig.WEAVIATE_PORT == 8080
    assert VectorDBConfig.MILVUS_PORT == 19530
    assert "http://" in VectorDBConfig.CHROMA_URL
    assert "http://" in VectorDBConfig.QDRANT_URL
    assert "http://" in VectorDBConfig.WEAVIATE_URL


@patch("chromadb.HttpClient")
def test_chroma_connection_check_success(mock_chroma):
    """Test successful connection check for Chroma."""
    mock_instance = MagicMock()
    mock_instance.heartbeat.return_value = 123456
    mock_chroma.return_value = mock_instance

    results = VectorDBManager.check_all_connections(quiet=True)
    assert "ChromaDB" in results
    assert results["ChromaDB"] is True
