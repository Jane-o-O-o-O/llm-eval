"""Tests for metrics create command."""

from __future__ import annotations

import os
import tempfile

from click.testing import CliRunner

from llm_eval.cli import main


class TestMetricsCreate:
    """Test scaffold for custom metric templates."""

    def test_creates_metric_file(self):
        """metrics create should generate a Python file."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "my_metric.py")
            result = runner.invoke(main, ["metrics", "create", "my_metric", "--output", output])
            assert result.exit_code == 0
            assert os.path.exists(output)
            with open(output) as f:
                content = f.read()
            assert "MyMetricMetric" in content or "class MyMetric" in content
            assert "evaluate" in content
            assert "MetricResult" in content

    def test_creates_with_default_output(self):
        """metrics create should default to metric name in current dir."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            result = runner.invoke(main, ["metrics", "create", "custom_score"])
            assert result.exit_code == 0
            expected = os.path.join(tmpdir, "custom_score_metric.py")
            assert os.path.exists(expected)

    def test_create_help(self):
        """metrics create --help should show usage."""
        runner = CliRunner()
        result = runner.invoke(main, ["metrics", "create", "--help"])
        assert result.exit_code == 0
        assert "NAME" in result.output or "name" in result.output.lower()
