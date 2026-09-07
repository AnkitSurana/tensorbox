"""Unified diagnostic and health-check utility for all Tensorbox services."""
import socket
import sys
from utils.config import Config
from utils.logger import get_logger

logger = get_logger("HealthCheck")


def check_tcp_port(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a service port is actively accepting connections."""
    target_hosts = [host]
    if host in ("postgres", "redis", "chromadb", "localhost", "") and "127.0.0.1" not in target_hosts:
        target_hosts.append("127.0.0.1")

    for target in target_hosts:
        try:
            with socket.create_connection((target, port), timeout=timeout):
                return True
        except (socket.timeout, ConnectionRefusedError, OSError):
            continue
    return False


def check_all_services(verbose: bool = True):
    """Diagnose and log the availability of all Tensorbox services."""
    services = {
        # Core & Local Services
        "PostgreSQL": {"host": Config.get("POSTGRES_HOST", "127.0.0.1"), "port": int(Config.get("POSTGRES_PORT", "5432")), "start_cmd": "tensorbox postgres"},
        "Redis": {"host": Config.get("REDIS_HOST", "127.0.0.1"), "port": int(Config.get("REDIS_PORT", "6379")), "start_cmd": "tensorbox redis"},
        "ChromaDB": {"host": Config.get("CHROMA_HOST", "127.0.0.1"), "port": int(Config.get("CHROMA_PORT", "8000")), "start_cmd": "tensorbox chroma"},
        "Streamlit": {"host": "127.0.0.1", "port": 8501, "start_cmd": "tensorbox streamlit"},
        "FastAPI": {"host": "127.0.0.1", "port": 5000, "start_cmd": "tensorbox fastapi"},
        "TensorBoard": {"host": "127.0.0.1", "port": 6006, "start_cmd": "tensorbox tensorboard"},
    }

    if verbose:
        print("\n" + "=" * 65)
        print("🔍 TENSORBOX SERVICES STATUS & HEALTH LOGS")
        print("=" * 65)

    status_report = {}
    for name, info in services.items():
        is_up = check_tcp_port(info["host"], info["port"])
        status_report[name] = is_up
        if is_up:
            logger.info(f"✅ {name:<12} [ONLINE]  (Port: {info['port']})")
        else:
            logger.info(f"⏸️  {name:<12} [OFFLINE] -> Launch via: {info['start_cmd']}")

    if verbose:
        print("\n🔑 LLM Provider API Keys:")
        keys = Config.validate_llm_keys()
        for provider, configured in keys.items():
            if configured:
                logger.info(f"   ✓ {provider.capitalize():<12}: Configured in .env")
            else:
                logger.info(f"   • {provider.capitalize():<12}: Not set -> Run: tensorbox set-key {provider.lower()} <your-key>")
        print("=" * 65 + "\n")

    return status_report



if __name__ == "__main__":
    check_all_services(verbose=True)
