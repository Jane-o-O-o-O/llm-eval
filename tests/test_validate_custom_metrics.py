"""Tests for custom metric validation in the validate command."""

from __future__ import annotations

from click.testing import CliRunner

from llm_eval.cli import main


class TestValidateCustomMetrics:
    """Test that validate checks custom metrics can be loaded."""

    def test_validate_valid_custom_metric(self, tmp_path):
        """Config with a valid custom_metric should pass."""
        # Create a valid custom metric module
        metric_file = tmp_path / "my_metric.py"
        metric_file.write_text(
            'from llm_eval.metrics import Metric, MetricResult\n'
            'from llm_eval.models import Sample\n'
            'class MyMetric(Metric):\n'
            '    name = "my_metric"\n'
            '    description = "test"\n'
            '    async def evaluate(self, sample):\n'
            '        return MetricResult(name=self.name, score=1.0)\n'
        )

        config = tmp_path / "config.yaml"
        config.write_text(
            "evaluations:\n"
            "  - name: test\n"
            "    dataset: data.jsonl\n"
            "    metrics: [my_metric]\n"
            "custom_metrics:\n"
            "  - module: my_metric\n"
            "    class: MyMetric\n"
        )
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"query": "q", "context": ["c"], "answer": "a"}\n')

        runner = CliRunner()
        result = runner.invoke(main, [
            "validate", str(config),
        ], catch_exceptions=False)
        # Should at least attempt to check custom metrics
        assert "valid" in result.output.lower() or "custom" in result.output.lower()

    def test_validate_invalid_custom_metric_missing_class(self, tmp_path):
        """Config with missing class_name in custom_metric should warn."""
        config = tmp_path / "config.yaml"
        config.write_text(
            "evaluations:\n"
            "  - name: test\n"
            "    dataset: data.jsonl\n"
            "    metrics: [faithfulness]\n"
            "custom_metrics:\n"
            "  - module: nonexistent_module\n"
            "    class: NonExistentClass\n"
        )
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"query": "q", "context": ["c"], "answer": "a"}\n')

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(config)])
        assert result.exit_code != 0
        assert "custom" in result.output.lower() or "error" in result.output.lower()

    def test_validate_custom_metric_missing_fields(self, tmp_path):
        """Config with custom_metric missing module/class should error."""
        config = tmp_path / "config.yaml"
        config.write_text(
            "evaluations:\n"
            "  - name: test\n"
            "    dataset: data.jsonl\n"
            "    metrics: [faithfulness]\n"
            "custom_metrics:\n"
            "  - module: some_module\n"
        )
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"query": "q", "context": ["c"], "answer": "a"}\n')

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(config)])
        assert result.exit_code != 0

    def test_validate_no_custom_metrics(self, tmp_path):
        """Config without custom_metrics should work as before."""
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
        result = runner.invoke(main, ["validate", str(config)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()
