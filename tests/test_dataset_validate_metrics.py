"""Tests for improved dataset validate with metric-specific checks."""

from __future__ import annotations

from click.testing import CliRunner

from llm_eval.cli import main


class TestDatasetValidateMetrics:
    """Test metric-specific dataset validation."""

    def test_validate_warns_missing_reference_for_similarity(self, tmp_path):
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"query": "q", "context": ["c"], "answer": "a"}\n')

        runner = CliRunner()
        result = runner.invoke(main, [
            "dataset", "validate", str(ds),
            "--metrics", "answer_similarity",
        ])
        assert result.exit_code == 0
        assert "reference" in result.output.lower()

    def test_validate_warns_missing_context_for_faithfulness(self, tmp_path):
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"query": "q", "context": [], "answer": "a"}\n')

        runner = CliRunner()
        result = runner.invoke(main, [
            "dataset", "validate", str(ds),
            "--metrics", "faithfulness",
        ])
        assert result.exit_code == 0
        assert "context" in result.output.lower()

    def test_validate_no_warning_when_all_fields_present(self, tmp_path):
        ds = tmp_path / "data.jsonl"
        ds.write_text(
            '{"query": "q", "context": ["c"], "answer": "a", "reference": "r"}\n'
        )

        runner = CliRunner()
        result = runner.invoke(main, [
            "dataset", "validate", str(ds),
            "--metrics", "answer_similarity,faithfulness",
        ])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_without_metrics_flag(self, tmp_path):
        """Without --metrics, only basic checks are performed (backward compat)."""
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"query": "q", "context": [], "answer": "a"}\n')

        runner = CliRunner()
        result = runner.invoke(main, ["dataset", "validate", str(ds)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_validate_warns_missing_context_for_context_recall(self, tmp_path):
        ds = tmp_path / "data.jsonl"
        ds.write_text(
            '{"query": "q", "context": [], "answer": "a", "reference": "r"}\n'
        )

        runner = CliRunner()
        result = runner.invoke(main, [
            "dataset", "validate", str(ds),
            "--metrics", "context_recall",
        ])
        assert result.exit_code == 0
        assert "context" in result.output.lower()

    def test_validate_multiple_metrics(self, tmp_path):
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"query": "q", "context": ["c"], "answer": "a"}\n')

        runner = CliRunner()
        result = runner.invoke(main, [
            "dataset", "validate", str(ds),
            "--metrics", "answer_similarity,context_recall",
        ])
        assert result.exit_code == 0
        assert "reference" in result.output.lower()
