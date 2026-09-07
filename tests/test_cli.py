"""Tests for Tensorbox CLI module."""
from unittest.mock import patch
from pathlib import Path
from utils.cli import cmd_set_key, cmd_help, cmd_new
from utils.config import Config


def test_cmd_help():
    """Verify help command runs without errors."""
    cmd_help()


def test_cmd_set_key_updates_config():
    """Verify set-key command updates the active configuration."""
    with patch("utils.config.dotenv_set_key"):
        cmd_set_key(["openai", "sk-test-key-12345"])
        assert Config.get("OPENAI_API_KEY") == "sk-test-key-12345"


def test_cmd_set_key_missing_args(capsys):
    """Verify set-key with missing arguments prints usage."""
    cmd_set_key(["openai"])
    captured = capsys.readouterr()
    assert "tensorbox set-key" in captured.out or "Usage:" in captured.out


def test_cmd_new_scaffolding(tmp_path, monkeypatch):
    """Verify tensorbox new creates starter template files."""
    monkeypatch.chdir(tmp_path)
    cmd_new(["test-assignment"])
    created_dir = tmp_path / "projects" / "test-assignment"
    assert created_dir.is_dir()
    assert (created_dir / "main.py").is_file()
    assert (created_dir / "README.md").is_file()
    assert (created_dir / "test_main.py").is_file()


def test_cmd_git_init_creates_secure_gitignore(tmp_path, monkeypatch):
    """Verify tensorbox git-init generates secure exclusion rules."""
    from utils.cli import cmd_git_init
    monkeypatch.chdir(tmp_path)
    cmd_git_init()
    gitignore = tmp_path / ".gitignore"
    assert gitignore.is_file()
    content = gitignore.read_text()
    assert ".env" in content
    assert "data/" in content
    assert "models/" in content
    assert "logs/" in content

