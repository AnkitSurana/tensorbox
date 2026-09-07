"""Vector Database utilities and connection verification."""
import os
from dotenv import load_dotenv

load_dotenv()


class VectorDBConfig:
    """Configuration for vector databases."""

    CHROMA_HOST = os.getenv("CHROMA_HOST", "127.0.0.1")
    CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_URL = f"http://{CHROMA_HOST}:{CHROMA_PORT}"

    QDRANT_HOST = os.getenv("QDRANT_HOST", "127.0.0.1")
    QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_URL = f"http://{QDRANT_HOST}:{QDRANT_PORT}"

    WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "127.0.0.1")
    WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
    WEAVIATE_URL = f"http://{WEAVIATE_HOST}:{WEAVIATE_PORT}"

    MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
    MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))


class VectorDBManager:
    """Unified vector database interface and connection verifier."""

    @staticmethod
    def check_all_connections(quiet=False):
        """Check availability of configured vector databases."""
        if not quiet:
            print("\n" + "=" * 50)
            print("VECTOR DATABASE CONNECTION CHECK")
            print("=" * 50 + "\n")

        results = {}

        # ChromaDB
        try:
            import chromadb
            client = chromadb.HttpClient(
                host=VectorDBConfig.CHROMA_HOST,
                port=VectorDBConfig.CHROMA_PORT
            )
            client.heartbeat()
            results["ChromaDB"] = True
            if not quiet:
                print("✅ ChromaDB (Running)")
        except Exception as e:
            results["ChromaDB"] = False
            if not quiet:
                print(f"❌ ChromaDB: {str(e)[:50]}")

        # Qdrant
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(
                host=VectorDBConfig.QDRANT_HOST,
                port=VectorDBConfig.QDRANT_PORT,
                timeout=5
            )
            client.get_collections()
            results["Qdrant"] = True
            if not quiet:
                print("✅ Qdrant (Running)")
        except Exception as e:
            results["Qdrant"] = False
            if not quiet:
                print(f"❌ Qdrant: {str(e)[:50]}")

        # Weaviate
        try:
            import weaviate
            client = weaviate.Client(
                url=VectorDBConfig.WEAVIATE_URL,
                timeout_config=(3, 5)
            )
            if client.is_ready():
                results["Weaviate"] = True
                if not quiet:
                    print("✅ Weaviate (Running)")
            else:
                results["Weaviate"] = False
                if not quiet:
                    print("❌ Weaviate: Not ready")
        except Exception as e:
            results["Weaviate"] = False
            if not quiet:
                print(f"❌ Weaviate: {str(e)[:50]}")

        # Milvus
        try:
            from pymilvus import connections
            connections.connect(
                "default",
                host=VectorDBConfig.MILVUS_HOST,
                port=VectorDBConfig.MILVUS_PORT,
                timeout=5
            )
            results["Milvus"] = True
            if not quiet:
                print("✅ Milvus (Running)")
        except Exception as e:
            results["Milvus"] = False
            if not quiet:
                print(f"❌ Milvus: {str(e)[:50]}")

        if not quiet:
            print("\n" + "=" * 50 + "\n")
        return results
