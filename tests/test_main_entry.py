"""Tests for __main__.py module entry point."""

from __future__ import annotations

import subprocess
import sys


class TestMainEntryPoint:
    """Tests for python -m llm_eval."""

    def test_module_entry_shows_help(self) -> None:
        """Test that python -m llm_eval --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "llm_eval", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "init" in result.stdout or "run" in result.stdout

    def test_module_entry_lists_metrics(self) -> None:
        """Test that python -m llm_eval metrics works."""
        result = subprocess.run(
            [sys.executable, "-m", "llm_eval", "metrics"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "faithfulness" in result.stdout
        assert "Total:" in result.stdout

    def test_module_entry_run_help(self) -> None:
        """Test that python -m llm_eval run --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "llm_eval", "run", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout

    def test_module_entry_init_help(self) -> None:
        """Test that python -m llm_eval init --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "llm_eval", "init", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--output" in result.stdout
