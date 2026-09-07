"""Tests for logger and health check utilities."""
from utils.logger import get_logger
from utils.health_check import check_tcp_port, check_all_services


def test_logger_creation():
    """Verify custom logger creation and level."""
    log = get_logger("test-logger")
    assert log is not None
    assert log.name == "test-logger"


def test_check_tcp_port_offline():
    """Verify offline port check returns False without raising unhandled errors."""
    result = check_tcp_port("127.0.0.1", 65432, timeout=0.1)
    assert result is False


def test_check_all_services():
    """Verify check_all_services returns a dictionary of service statuses."""
    report = check_all_services(verbose=False)
    assert isinstance(report, dict)
    assert "ChromaDB" in report
    assert "PostgreSQL" in report
    assert "Redis" in report
    assert "FastAPI" in report
    assert "Streamlit" in report
