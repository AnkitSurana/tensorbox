"""Configuration management for AIML devbox."""
import os
from pathlib import Path
from dotenv import load_dotenv, set_key as dotenv_set_key

_loaded = False


def find_env_file() -> Path:
    """Find the active .env file location."""
    candidates = [
        Path("/workspace/.env"),
        Path.cwd() / ".env",
        Path(__file__).parent.parent / ".env",
        Path("/opt/devbox/.env"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path("/workspace/.env") if Path("/workspace").exists() else Path.cwd() / ".env"


def reload_config(force: bool = False):
    """Reload environment variables from .env file."""
    global _loaded
    if _loaded and not force:
        return
    env_file = find_env_file()
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=False)
    else:
        load_dotenv(override=False)
    _loaded = True


reload_config()


class Config:
    @classmethod
    def get_api_key(cls, provider: str) -> str:
        return os.getenv(f"{provider.upper()}_API_KEY", "")

    @classmethod
    def has_key(cls, provider: str) -> bool:
        val = cls.get_api_key(provider)
        return bool(val and not val.startswith("your-") and not val.startswith("sk-your-") and not val.startswith("hf_your-"))

    """Central configuration loader."""

    @classmethod
    def get(cls, key: str, default: str = "") -> str:
        return os.getenv(key, default)

    @classmethod
    def set(cls, key: str, value: str):
        """Persist a key-value pair to the .env file and update environment."""
        env_file = find_env_file()
        if not env_file.exists():
            env_file.touch()
        try:
            dotenv_set_key(str(env_file), key, value)
        except Exception:
            pass
        os.environ[key] = value

    @classmethod
    def OPENAI_API_KEY(cls):
        return cls.get("OPENAI_API_KEY", "")

    # Vector Databases
    CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
    QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")
    WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
    MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))

    # Relational & NoSQL Databases
    POSTGRES_HOST = os.getenv("POSTGRES_HOST", "127.0.0.1")
    POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
    POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")
    POSTGRES_DB = os.getenv("POSTGRES_DB", "ml_database")

    MONGO_HOST = os.getenv("MONGO_HOST", "127.0.0.1")
    MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
    MONGO_USER = os.getenv("MONGO_USER", "admin")
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "admin")

    REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
    REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "redis-password")

    # App Settings
    ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
    DEBUG = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate_llm_keys(cls):
        """Check which LLM providers have API keys configured."""
        openai = cls.get("OPENAI_API_KEY")
        anthropic = cls.get("ANTHROPIC_API_KEY")
        google = cls.get("GOOGLE_API_KEY")
        cohere = cls.get("COHERE_API_KEY")
        hf = cls.get("HUGGINGFACE_API_KEY")

        return {
            "openai": bool(openai and not openai.startswith("sk-your-")),
            "anthropic": bool(anthropic and not anthropic.startswith("sk-ant-your-")),
            "google": bool(google and not google.startswith("your-")),
            "cohere": bool(cohere and not cohere.startswith("your-")),
            "huggingface": bool(hf and not hf.startswith("hf_your-")),
        }

# Singleton instance for convenience
config = Config
