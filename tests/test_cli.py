"""Tests for the CLI commands."""

import json
import os

import pytest
from click.testing import CliRunner
from llm_eval.cli import main


class TestInitCommand:
    """Tests for the `llm-eval init` command."""

    def test_init_creates_config_file(self, tmp_path) -> None:
        runner = CliRunner()
        output = str(tmp_path / "evals.yaml")
        result = runner.invoke(main, ["init", "--output", output])
        assert result.exit_code == 0
        assert os.path.exists(output)

    def test_init_output_contains_judge_section(self, tmp_path) -> None:
        runner = CliRunner()
        output = str(tmp_path / "evals.yaml")
        runner.invoke(main, ["init", "--output", output])
        with open(output, "r") as f:
            content = f.read()
        assert "judge:" in content
        assert "evaluations:" in content

    def test_init_creates_sample_dataset(self, tmp_path) -> None:
        runner = CliRunner()
        output = str(tmp_path / "evals.yaml")
        runner.invoke(main, ["init", "--output", output])
        sample_path = tmp_path / "samples.jsonl"
        assert sample_path.exists()
        with open(sample_path, "r") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) >= 1

    def test_init_default_output(self, tmp_path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0
            assert os.path.exists("evals.yaml")


class TestRunCommand:
    """Tests for the `llm-eval run` command."""

    def test_run_missing_config_exits_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--config", "/nonexistent/config.yaml"])
        assert result.exit_code != 0

    def test_run_with_config_and_dataset(self, tmp_path) -> None:
        """Test run with a config file pointing to a dataset."""
        import yaml

        # Create dataset
        dataset = [
            {
                "query": "What is Python?",
                "context": ["Python is a programming language."],
                "answer": "Python is a programming language.",
            }
        ]
        dataset_path = tmp_path / "samples.jsonl"
        dataset_path.write_text("\n".join(json.dumps(d) for d in dataset) + "\n")

        # Create config
        config = {
            "judge": {"model": "gpt-4o"},
            "evaluations": [
                {
                    "name": "Test Eval",
                    "type": "rag",
                    "dataset": str(dataset_path),
                    "metrics": ["faithfulness"],
                }
            ],
            "defaults": {"threshold": 0.7},
        }
        config_path = tmp_path / "evals.yaml"
        config_path.write_text(yaml.dump(config))

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--config", str(config_path)])
        # Should fail because no API key, but config parsing should work
        assert result.exit_code != 0

    def test_run_shows_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_main_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "run" in result.output


class TestReportGeneration:
    """Tests for the report/output generation."""

    def test_terminal_report(self) -> None:
        from llm_eval.report import format_terminal_report
        from llm_eval.models import EvalResult, MetricResult

        results = [
            EvalResult(
                sample_index=0,
                metrics=[MetricResult(name="faithfulness", score=0.95, details={})],
            ),
        ]
        summary = {
            "total_samples": 1,
            "overall_score": 0.95,
            "pass_count": 1,
            "fail_count": 0,
            "pass_rate": 1.0,
            "threshold": 0.7,
            "metric_scores": {
                "faithfulness": {"mean": 0.95, "min": 0.95, "max": 0.95},
            },
        }
        output = format_terminal_report(results, summary)
        assert "faithfulness" in output
        assert "0.95" in output
        assert "PASS" in output

    def test_json_report(self) -> None:
        from llm_eval.report import format_json_report
        from llm_eval.models import EvalResult, MetricResult

        results = [
            EvalResult(
                sample_index=0,
                metrics=[MetricResult(name="faithfulness", score=0.8, details={})],
            ),
        ]
        summary = {
            "total_samples": 1,
            "overall_score": 0.8,
            "pass_count": 1,
            "fail_count": 0,
            "pass_rate": 1.0,
            "threshold": 0.7,
            "metric_scores": {},
        }
        output = format_json_report(results, summary)
        parsed = json.loads(output)
        assert parsed["summary"]["total_samples"] == 1
        assert len(parsed["results"]) == 1

    def test_csv_report(self) -> None:
        from llm_eval.report import format_csv_report
        from llm_eval.models import EvalResult, MetricResult

        results = [
            EvalResult(
                sample_index=0,
                metrics=[MetricResult(name="faithfulness", score=0.85, details={})],
            ),
            EvalResult(
                sample_index=1,
                metrics=[MetricResult(name="faithfulness", score=0.6, details={})],
            ),
        ]
        output = format_csv_report(results)
        lines = output.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert "sample_index" in lines[0]
        assert "faithfulness" in lines[0]
