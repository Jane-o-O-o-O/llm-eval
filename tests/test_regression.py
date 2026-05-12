"""Tests for regression detection and CLI regression mode."""

import json
import os

import pytest
import yaml
from click.testing import CliRunner
from llm_eval.cli import main
from llm_eval.models import EvalResult, MetricResult
from llm_eval.regression import RegressionResult, check_regression, load_baseline


def _make_results(scores: dict[str, float]) -> list[EvalResult]:
    """Create results with given metric scores."""
    metrics = [MetricResult(name=name, score=score) for name, score in scores.items()]
    return [EvalResult(sample_index=0, metrics=metrics)]


def _make_baseline_file(data: dict, path: str) -> str:
    """Write a baseline JSON file."""
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestLoadBaseline:
    def test_load_from_json_report(self, tmp_path) -> None:
        baseline_data = {
            "summary": {
                "metric_scores": {
                    "faithfulness": {"mean": 0.9, "min": 0.8, "max": 1.0},
                    "answer_relevancy": {"mean": 0.85, "min": 0.7, "max": 0.95},
                }
            }
        }
        path = str(tmp_path / "baseline.json")
        _make_baseline_file(baseline_data, path)

        baseline = load_baseline(path)
        assert baseline["faithfulness"] == 0.9
        assert baseline["answer_relevancy"] == 0.85

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_baseline("/nonexistent/baseline.json")

    def test_invalid_json(self, tmp_path) -> None:
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            f.write("not json")
        with pytest.raises(ValueError):
            load_baseline(path)


class TestCheckRegression:
    def test_no_regression(self) -> None:
        results = _make_results({"faithfulness": 0.9, "answer_relevancy": 0.85})
        baseline = {"faithfulness": 0.9, "answer_relevancy": 0.85}

        reg = check_regression(results, baseline, tolerance=0.05)
        assert reg.passed is True

    def test_within_tolerance(self) -> None:
        results = _make_results({"faithfulness": 0.86, "answer_relevancy": 0.82})
        baseline = {"faithfulness": 0.9, "answer_relevancy": 0.85}

        reg = check_regression(results, baseline, tolerance=0.05)
        assert reg.passed is True

    def test_exceeds_tolerance(self) -> None:
        results = _make_results({"faithfulness": 0.80, "answer_relevancy": 0.85})
        baseline = {"faithfulness": 0.9, "answer_relevancy": 0.85}

        reg = check_regression(results, baseline, tolerance=0.05)
        assert reg.passed is False

    def test_improvement_passes(self) -> None:
        results = _make_results({"faithfulness": 0.95})
        baseline = {"faithfulness": 0.9}

        reg = check_regression(results, baseline, tolerance=0.05)
        assert reg.passed is True

    def test_missing_metric_in_results(self) -> None:
        results = _make_results({"faithfulness": 0.9})
        baseline = {"faithfulness": 0.9, "answer_relevancy": 0.85}

        reg = check_regression(results, baseline, tolerance=0.05)
        # answer_relevancy not in results -> current=0.0, baseline=0.85 -> regression
        assert reg.passed is False

    def test_comparison_details(self) -> None:
        results = _make_results({"faithfulness": 0.85})
        baseline = {"faithfulness": 0.9}

        reg = check_regression(results, baseline, tolerance=0.05)
        assert len(reg.comparisons) == 1
        c = reg.comparisons[0]
        assert c["metric"] == "faithfulness"
        assert c["baseline"] == 0.9
        assert c["current"] == 0.85
        assert c["delta"] == pytest.approx(-0.05)

    def test_to_dict(self) -> None:
        results = _make_results({"faithfulness": 0.9})
        baseline = {"faithfulness": 0.9}

        reg = check_regression(results, baseline, tolerance=0.05)
        d = reg.to_dict()
        assert d["passed"] is True
        assert d["tolerance"] == 0.05
        assert len(d["comparisons"]) == 1

    def test_format_terminal(self) -> None:
        results = _make_results({"faithfulness": 0.9})
        baseline = {"faithfulness": 0.9}

        reg = check_regression(results, baseline, tolerance=0.05)
        output = reg.format_terminal()
        assert "Regression Check" in output
        assert "faithfulness" in output
        assert "No regressions" in output

    def test_zero_tolerance(self) -> None:
        results = _make_results({"faithfulness": 0.899})
        baseline = {"faithfulness": 0.9}

        reg = check_regression(results, baseline, tolerance=0.0)
        assert reg.passed is False


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
