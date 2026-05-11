"""Tests for regression detection."""

import json
import os
import tempfile

import pytest
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
