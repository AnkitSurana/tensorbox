"""Centralized logging utility for Tensorbox with file and console handlers."""
import sys
import os
import logging
from pathlib import Path


def get_log_dir() -> Path:
    """Determine active log directory."""
    candidates = [
        Path("/workspace/logs"),
        Path.cwd() / "logs",
        Path(__file__).parent.parent / "logs",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Default to /workspace/logs or local logs
    target = Path("/workspace/logs") if Path("/workspace").exists() else Path.cwd() / "logs"
    try:
        target.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return target


def get_logger(name: str = "tensorbox") -> logging.Logger:
    """Create and configure a standardized logger with console & file outputs."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    level = logging.INFO
    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Console Stream Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 2. Persistent File Handler in /workspace/logs/tensorbox.log
    log_dir = get_log_dir()
    if log_dir.exists():
        try:
            file_handler = logging.FileHandler(log_dir / "tensorbox.log", encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass

    return logger


# Default logger instance
logger = get_logger("tensorbox")

