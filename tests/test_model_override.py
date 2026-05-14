"""Tests for --model CLI override."""

from __future__ import annotations

from click.testing import CliRunner

from llm_eval.cli import main


class TestModelOverride:
    """Test the --model CLI option on the run command."""

    def test_run_help_shows_model_option(self):
        """--model option should appear in run --help."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output

    def test_validate_with_model_override_in_help(self):
        """run --help should document --model."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert "judge model" in result.output.lower() or "model" in result.output.lower()
