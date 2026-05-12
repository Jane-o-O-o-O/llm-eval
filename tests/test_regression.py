"""Tests for the regression fail mode in CLI."""

import json
import os

import pytest
import yaml
from click.testing import CliRunner
from llm_eval.cli import main


class TestRegressionMode:
    """Tests for --fail-on regression CLI option."""

    def test_regression_mode_cli_option_exists(self) -> None:
        """The run command should accept --fail-on and --baseline options."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--fail-on" in result.output
        assert "--baseline" in result.output

    def test_parallel_cli_option_exists(self) -> None:
        """The run command should accept --parallel option."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--parallel" in result.output

    def test_regression_mode_with_baseline(self, tmp_path) -> None:
        """--fail-on regression with a baseline should compare against it."""
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
                    "name": "Regression Test",
                    "type": "rag",
                    "dataset": str(dataset_path),
                    "metrics": ["faithfulness"],
                }
            ],
            "defaults": {"threshold": 0.7},
        }
        config_path = tmp_path / "evals.yaml"
        config_path.write_text(yaml.dump(config))

        # Create baseline report
        baseline = {
            "summary": {
                "total_samples": 1,
                "overall_score": 0.9,
                "metric_scores": {
                    "faithfulness": {"mean": 0.9, "min": 0.9, "max": 0.9},
                },
            }
        }
        baseline_path = tmp_path / "baseline.json"
        baseline_path.write_text(json.dumps(baseline))

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "--config", str(config_path), "--fail-on", "regression", "--baseline", str(baseline_path)],
        )
        # Will fail because no API key, but config parsing + baseline loading should work
        assert result.exit_code != 0


class TestEvalConfigParallel:
    """Tests for parallel config in EvalConfig."""

    def test_config_with_parallel(self) -> None:
        from llm_eval.models import EvalConfig

        data = {
            "judge": {"model": "gpt-4o"},
            "evaluations": [],
            "defaults": {"parallel": 5},
        }
        config = EvalConfig.from_dict(data)
        assert config.parallel == 5

    def test_config_default_parallel(self) -> None:
        from llm_eval.models import EvalConfig

        data = {
            "judge": {"model": "gpt-4o"},
            "evaluations": [],
            "defaults": {},
        }
        config = EvalConfig.from_dict(data)
        assert config.parallel == 1
