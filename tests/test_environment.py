"""Tests for workspace structure and environment setup."""
from pathlib import Path
import pytest


def get_repo_root():
    """Locate repository root directory."""
    candidates = [
        Path(__file__).parent.parent,
        Path.cwd(),
        Path("/workspace")
    ]
    for c in candidates:
        if (c / "Dockerfile").exists():
            return c
    return None


def test_directory_structure():
    """Verify standard directories exist."""
    base = get_repo_root()
    if not base:
        pytest.skip("Repository root not present in standalone container image")

    expected_dirs = [
        "notebooks",
        "data",
        "datasets",
        "projects",
        "models",
        "utils",
        "tests",
        "docs",
        "scripts"
    ]
    for d in expected_dirs:
        dir_path = base / d
        assert dir_path.is_dir(), f"Expected directory {d} to exist"


def test_essential_files():
    """Verify essential config and docker files exist."""
    base = get_repo_root()
    if not base:
        pytest.skip("Repository root not present in standalone container image")

    expected_files = [
        "Dockerfile",
        "docker-compose.yml",
        ".env.example",
        ".gitignore",
        ".dockerignore",
        "requirements.txt",
        "requirements.in",
        "setup.py",
        "setup.sh",
        "README.md",
        "LICENSE"
    ]
    for f in expected_files:
        file_path = base / f
        assert file_path.is_file(), f"Expected file {f} to exist"


def test_documentation_structure():
    """Verify all documentation guides are consolidated in docs/."""
    base = get_repo_root()
    if not base:
        pytest.skip("Repository root not present in standalone container image")

    docs_dir = base / "docs"
    expected_docs = [
        "USER_GUIDE.md",
        "DEVELOPER_GUIDE.md",
        "COMMANDS.md",
        "INSTALL.md",
        "CONTRIBUTING.md",
        "SECURITY.md"
    ]
    for doc in expected_docs:
        doc_path = docs_dir / doc
        assert doc_path.is_file(), f"Expected doc file {doc} in docs/ folder"
