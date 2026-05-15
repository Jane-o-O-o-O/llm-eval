"""Tests for the export command — convert report files between formats."""

from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

from llm_eval.cli import main


SAMPLE_REPORT = {
    "metadata": {
        "timestamp": "2026-05-15T10:00:00Z",
        "version": "1.0.0",
    },
    "summary": {
        "total_samples": 2,
        "overall_score": 0.85,
        "pass_count": 2,
        "fail_count": 0,
        "pass_rate": 1.0,
        "threshold": 0.7,
        "metric_scores": {
            "faithfulness": {"mean": 0.9, "min": 0.8, "max": 1.0},
            "answer_relevancy": {"mean": 0.8, "min": 0.7, "max": 0.9},
        },
    },
    "results": [
        {
            "sample_index": 0,
            "overall_score": 0.85,
            "metrics": [
                {"name": "faithfulness", "score": 0.9, "details": {}},
                {"name": "answer_relevancy", "score": 0.8, "details": {}},
            ],
        },
        {
            "sample_index": 1,
            "overall_score": 0.85,
            "metrics": [
                {"name": "faithfulness", "score": 0.8, "details": {}},
                {"name": "answer_relevancy", "score": 0.9, "details": {}},
            ],
        },
    ],
}


@pytest.fixture
def runner():
    return CliRunner()


class TestExportCommand:
    """Test the export command for converting report formats."""

    def test_export_json_to_html(self, tmp_path, runner):
        """Export JSON report to HTML."""
        input_path = str(tmp_path / "report.json")
        output_path = str(tmp_path / "report.html")

        with open(input_path, "w") as f:
            json.dump(SAMPLE_REPORT, f)

        result = runner.invoke(main, ["export", input_path, "--to", "html", "-o", output_path])
        assert result.exit_code == 0
        assert os.path.exists(output_path)

        with open(output_path) as f:
            content = f.read()
        assert "<html" in content

    def test_export_json_to_csv(self, tmp_path, runner):
        """Export JSON report to CSV."""
        input_path = str(tmp_path / "report.json")
        output_path = str(tmp_path / "report.csv")

        with open(input_path, "w") as f:
            json.dump(SAMPLE_REPORT, f)

        result = runner.invoke(main, ["export", input_path, "--to", "csv", "-o", output_path])
        assert result.exit_code == 0
        assert os.path.exists(output_path)

        with open(output_path) as f:
            content = f.read()
        assert "sample_index" in content

    def test_export_json_to_markdown(self, tmp_path, runner):
        """Export JSON report to Markdown."""
        input_path = str(tmp_path / "report.json")
        output_path = str(tmp_path / "report.md")

        with open(input_path, "w") as f:
            json.dump(SAMPLE_REPORT, f)

        result = runner.invoke(main, ["export", input_path, "--to", "markdown", "-o", output_path])
        assert result.exit_code == 0
        assert os.path.exists(output_path)

    def test_export_json_to_junit(self, tmp_path, runner):
        """Export JSON report to JUnit XML."""
        input_path = str(tmp_path / "report.json")
        output_path = str(tmp_path / "report.xml")

        with open(input_path, "w") as f:
            json.dump(SAMPLE_REPORT, f)

        result = runner.invoke(main, ["export", input_path, "--to", "junit", "-o", output_path])
        assert result.exit_code == 0
        assert os.path.exists(output_path)

        with open(output_path) as f:
            content = f.read()
        assert "testsuite" in content

    def test_export_to_stdout(self, tmp_path, runner):
        """Export outputs to stdout when no -o specified."""
        input_path = str(tmp_path / "report.json")

        with open(input_path, "w") as f:
            json.dump(SAMPLE_REPORT, f)

        result = runner.invoke(main, ["export", input_path, "--to", "csv"])
        assert result.exit_code == 0
        assert "sample_index" in result.output

    def test_export_invalid_input(self, runner):
        """Export handles non-existent input file."""
        result = runner.invoke(main, ["export", "nonexistent.json", "--to", "html"])
        assert result.exit_code != 0

    def test_export_auto_detect_extension(self, tmp_path, runner):
        """Export auto-detects output format from extension when --to omitted."""
        input_path = str(tmp_path / "report.json")
        output_path = str(tmp_path / "report.html")

        with open(input_path, "w") as f:
            json.dump(SAMPLE_REPORT, f)

        result = runner.invoke(main, ["export", input_path, "-o", output_path])
        assert result.exit_code == 0
        assert os.path.exists(output_path)

        with open(output_path) as f:
            content = f.read()
        assert "<html" in content

    def test_export_terminal(self, tmp_path, runner):
        """Export to terminal format."""
        input_path = str(tmp_path / "report.json")

        with open(input_path, "w") as f:
            json.dump(SAMPLE_REPORT, f)

        result = runner.invoke(main, ["export", input_path, "--to", "terminal"])
        assert result.exit_code == 0
        assert "Evaluation Report" in result.output
