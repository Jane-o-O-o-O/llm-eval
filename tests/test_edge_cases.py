"""Tests for improved error messages and edge case handling."""

from __future__ import annotations

from click.testing import CliRunner

from llm_eval.cli import main


class TestEdgeCases:
    """Test CLI edge cases and error messages."""

    def test_regression_without_baseline(self, tmp_path):
        """--fail-on regression without --baseline should give clear error."""
        config = tmp_path / "config.yaml"
        config.write_text(
            "evaluations:\n"
            "  - name: test\n"
            "    dataset: data.jsonl\n"
            "    metrics: [faithfulness]\n"
        )
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"query": "q", "context": ["c"], "answer": "a"}\n')

        runner = CliRunner()
        result = runner.invoke(main, [
            "run", "--config", str(config),
            "--fail-on", "regression",
        ])
        assert result.exit_code != 0
        assert "baseline" in result.output.lower()

    def test_empty_dataset(self, tmp_path):
        """Empty dataset should give clear error."""
        config = tmp_path / "config.yaml"
        config.write_text(
            "evaluations:\n"
            "  - name: test\n"
            "    dataset: data.jsonl\n"
            "    metrics: [faithfulness]\n"
        )
        ds = tmp_path / "data.jsonl"
        ds.write_text("")

        runner = CliRunner()
        result = runner.invoke(main, [
            "run", "--config", str(config), "--dry-run",
        ])
        # Should handle empty dataset gracefully
        assert result.exit_code == 0 or "empty" in result.output.lower() or "0 samples" in result.output

    def test_validate_nonexistent_config(self):
        """Validating nonexistent file should give clear error."""
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "/nonexistent/config.yaml"])
        assert result.exit_code != 0

    def test_validate_non_dict_yaml(self, tmp_path):
        """YAML that's not a mapping should give clear error."""
        config = tmp_path / "config.yaml"
        config.write_text("- just a list\n- not a mapping\n")

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(config)])
        assert result.exit_code != 0
        assert "mapping" in result.output.lower() or "yaml" in result.output.lower()

    def test_compare_missing_file(self):
        """Compare with missing file should give clear error."""
        runner = CliRunner()
        result = runner.invoke(main, [
            "compare", "/nonexistent/a.json", "/nonexistent/b.json",
        ])
        assert result.exit_code != 0

    def test_set_with_no_config(self):
        """--set with nonexistent config should give clear error."""
        runner = CliRunner()
        result = runner.invoke(main, [
            "run", "--config", "/nonexistent/config.yaml",
            "--set", "judge.model=test",
        ])
        assert result.exit_code != 0
        assert "not found" in result.output.lower()

    def test_run_with_unknown_metric(self, tmp_path):
        """Config with unknown metric should error in dry-run."""
        config = tmp_path / "config.yaml"
        config.write_text(
            "evaluations:\n"
            "  - name: test\n"
            "    dataset: data.jsonl\n"
            "    metrics: [nonexistent_metric]\n"
        )
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"query": "q", "context": ["c"], "answer": "a"}\n')

        runner = CliRunner()
        result = runner.invoke(main, [
            "run", "--config", str(config), "--dry-run",
        ])
        assert "unknown" in result.output.lower() or "nonexistent" in result.output.lower()

    def test_dataset_info_empty(self, tmp_path):
        """dataset info on empty file should handle gracefully."""
        ds = tmp_path / "empty.jsonl"
        ds.write_text("")

        runner = CliRunner()
        result = runner.invoke(main, ["dataset", "info", str(ds)])
        # Should not crash
        assert result.exit_code == 0 or "error" in result.output.lower() or "empty" in result.output.lower()
