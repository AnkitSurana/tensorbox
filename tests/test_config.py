"""Tests for Configuration Loader."""
import os
import pytest
from utils.config import Config


def test_config_defaults():
    """Verify default configurations have valid fallback values."""
    assert Config.POSTGRES_PORT == 5432
    assert Config.CHROMA_PORT == 8000
    assert Config.QDRANT_PORT == 6333
    assert Config.REDIS_PORT == 6379
    assert Config.MONGO_PORT == 27017
    assert isinstance(Config.DEBUG, bool)


def test_validate_llm_keys_with_placeholders(monkeypatch):
    """Ensure placeholder keys are identified as not configured."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-your-openai-key-here")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-your-anthropic-key-here")
    monkeypatch.setenv("GOOGLE_API_KEY", "your-google-api-key-here")

    validation = Config.validate_llm_keys()
    assert validation["openai"] is False
    assert validation["anthropic"] is False
    assert validation["google"] is False


def test_validate_llm_keys_with_real_keys(monkeypatch):
    """Ensure real keys are recognized as configured."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-1234567890abcdef")
    validation = Config.validate_llm_keys()
    assert validation["openai"] is True
