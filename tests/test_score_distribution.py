"""Tests for score distribution statistics in evaluator summary."""

from __future__ import annotations

import pytest

from llm_eval.evaluator import Evaluator
from llm_eval.models import EvalResult, MetricResult, Sample


@pytest.fixture
def evaluator():
    return Evaluator(metrics=["faithfulness"], threshold=0.7)


class TestScoreDistribution:
    """Test enhanced summary statistics."""

    def test_summary_has_median(self, evaluator):
        """Summary should include median."""
        results = [
            EvalResult(sample_index=0, metrics=[MetricResult("faithfulness", 0.9)]),
            EvalResult(sample_index=1, metrics=[MetricResult("faithfulness", 0.5)]),
            EvalResult(sample_index=2, metrics=[MetricResult("faithfulness", 0.7)]),
        ]
        summary = evaluator.summarize(results)
        assert "median" in summary
        assert summary["median"] == pytest.approx(0.7)

    def test_summary_has_p25_p75(self, evaluator):
        """Summary should include 25th and 75th percentiles."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        results = [
            EvalResult(sample_index=i, metrics=[MetricResult("faithfulness", s)])
            for i, s in enumerate(scores)
        ]
        summary = evaluator.summarize(results)
        assert "p25" in summary
        assert "p75" in summary
        assert summary["p25"] < summary["median"]
        assert summary["p75"] > summary["median"]

    def test_summary_has_std_dev(self, evaluator):
        """Summary should include standard deviation."""
        results = [
            EvalResult(sample_index=0, metrics=[MetricResult("faithfulness", 0.8)]),
            EvalResult(sample_index=1, metrics=[MetricResult("faithfulness", 0.6)]),
        ]
        summary = evaluator.summarize(results)
        assert "std_dev" in summary
        assert summary["std_dev"] > 0

    def test_summary_has_min_max(self, evaluator):
        """Summary should include min and max scores."""
        results = [
            EvalResult(sample_index=0, metrics=[MetricResult("faithfulness", 0.9)]),
            EvalResult(sample_index=1, metrics=[MetricResult("faithfulness", 0.3)]),
        ]
        summary = evaluator.summarize(results)
        assert "min_score" in summary
        assert "max_score" in summary
        assert summary["min_score"] == pytest.approx(0.3)
        assert summary["max_score"] == pytest.approx(0.9)

    def test_empty_results_distribution(self, evaluator):
        """Empty results should have zero distribution stats."""
        results: list[EvalResult] = []
        summary = evaluator.summarize(results)
        assert summary.get("median", 0.0) == 0.0
        assert summary.get("p25", 0.0) == 0.0
        assert summary.get("p75", 0.0) == 0.0
        assert summary.get("std_dev", 0.0) == 0.0

    def test_single_result_distribution(self, evaluator):
        """Single result: median = score, std_dev = 0."""
        results = [
            EvalResult(sample_index=0, metrics=[MetricResult("faithfulness", 0.8)]),
        ]
        summary = evaluator.summarize(results)
        assert summary["median"] == pytest.approx(0.8)
        assert summary["std_dev"] == pytest.approx(0.0)
        assert summary["min_score"] == pytest.approx(0.8)
        assert summary["max_score"] == pytest.approx(0.8)
