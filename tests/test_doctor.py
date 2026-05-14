"""Tests for the doctor CLI command."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from llm_eval.cli import main


@pytest.fixture
def runner():
    """Click test runner."""
    return CliRunner()


class TestDoctorCommand:
    """Test the `llm-eval doctor` command."""

    def test_doctor_runs(self, runner):
        """Doctor command should run without errors."""
        result = runner.invoke(main, ["doctor"])
        assert result.exit_code == 0
        assert "llm-eval Doctor" in result.output

    def test_doctor_shows_python_version(self, runner):
        """Doctor should show Python version."""
        result = runner.invoke(main, ["doctor"])
        assert "Python" in result.output

    def test_doctor_shows_metrics(self, runner):
        """Doctor should list available metrics."""
        result = runner.invoke(main, ["doctor"])
        assert "metrics available" in result.output

    def test_doctor_shows_version(self, runner):
        """Doctor should show llm-eval version."""
        result = runner.invoke(main, ["doctor"])
        assert "llm-eval v" in result.output

    def test_doctor_shows_cache_info(self, runner):
        """Doctor should show cache information."""
        result = runner.invoke(main, ["doctor"])
        assert "Cache" in result.output

    def test_doctor_shows_deps(self, runner):
        """Doctor should check dependencies."""
        result = runner.invoke(main, ["doctor"])
        assert "click" in result.output
        assert "httpx" in result.output

    def test_doctor_with_api_key(self, runner, monkeypatch):
        """Doctor should show masked API key when set."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-1234567890abcdef")
        result = runner.invoke(main, ["doctor"])
        assert "OPENAI_API_KEY" in result.output
        assert "sk-12345" in result.output
        # Should not show the full key
        assert "sk-1234567890abcdef" not in result.output

    def test_doctor_without_api_keys(self, runner, monkeypatch):
        """Doctor should warn when no API keys are set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        result = runner.invoke(main, ["doctor"])
        assert "No API keys found" in result.output or "OPENAI_API_KEY" in result.output

    def test_doctor_shows_all_checks_passed(self, runner, monkeypatch):
        """Doctor should show all checks passed when everything is OK."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = runner.invoke(main, ["doctor"])
        assert "All checks passed" in result.output
