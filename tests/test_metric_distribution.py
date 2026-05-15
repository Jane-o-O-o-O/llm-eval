"""Tests for per-metric distribution statistics in evaluator summary."""

from __future__ import annotations

import pytest

from llm_eval.evaluator import Evaluator
from llm_eval.models import EvalResult, MetricResult


def _make_results() -> list[EvalResult]:
    """Create test results with known score distributions."""
    return [
        EvalResult(sample_index=0, metrics=[
            MetricResult(name="faithfulness", score=0.9),
            MetricResult(name="answer_relevancy", score=0.8),
        ]),
        EvalResult(sample_index=1, metrics=[
            MetricResult(name="faithfulness", score=0.7),
            MetricResult(name="answer_relevancy", score=0.6),
        ]),
        EvalResult(sample_index=2, metrics=[
            MetricResult(name="faithfulness", score=0.5),
            MetricResult(name="answer_relevancy", score=0.4),
        ]),
        EvalResult(sample_index=3, metrics=[
            MetricResult(name="faithfulness", score=0.3),
            MetricResult(name="answer_relevancy", score=0.2),
        ]),
    ]


class TestPerMetricDistribution:
    """Test that per-metric scores include distribution statistics."""

    def test_per_metric_has_median(self):
        evaluator = Evaluator(metrics=["faithfulness", "answer_relevancy"])
        results = _make_results()
        summary = evaluator.summarize(results)

        for metric_name in ("faithfulness", "answer_relevancy"):
            scores = summary["metric_scores"][metric_name]
            assert "median" in scores, f"{metric_name} missing median"

    def test_per_metric_has_p25_p75(self):
        evaluator = Evaluator(metrics=["faithfulness", "answer_relevancy"])
        results = _make_results()
        summary = evaluator.summarize(results)

        for metric_name in ("faithfulness", "answer_relevancy"):
            scores = summary["metric_scores"][metric_name]
            assert "p25" in scores, f"{metric_name} missing p25"
            assert "p75" in scores, f"{metric_name} missing p75"

    def test_per_metric_has_std_dev(self):
        evaluator = Evaluator(metrics=["faithfulness", "answer_relevancy"])
        results = _make_results()
        summary = evaluator.summarize(results)

        for metric_name in ("faithfulness", "answer_relevancy"):
            scores = summary["metric_scores"][metric_name]
            assert "std_dev" in scores, f"{metric_name} missing std_dev"

    def test_per_metric_median_value(self):
        evaluator = Evaluator(metrics=["faithfulness"])
        results = _make_results()
        summary = evaluator.summarize(results)

        # Faithfulness scores: 0.3, 0.5, 0.7, 0.9 → median = 0.6
        median = summary["metric_scores"]["faithfulness"]["median"]
        assert abs(median - 0.6) < 0.01

    def test_per_metric_p25_value(self):
        evaluator = Evaluator(metrics=["faithfulness"])
        results = _make_results()
        summary = evaluator.summarize(results)

        # Sorted: [0.3, 0.5, 0.7, 0.9], p25 = 0.35 (interpolated)
        p25 = summary["metric_scores"]["faithfulness"]["p25"]
        assert abs(p25 - 0.45) < 0.05  # Allow some interpolation tolerance

    def test_per_metric_std_dev_value(self):
        evaluator = Evaluator(metrics=["faithfulness"])
        results = _make_results()
        summary = evaluator.summarize(results)

        # Mean = 0.6, std_dev should be nonzero
        std_dev = summary["metric_scores"]["faithfulness"]["std_dev"]
        assert std_dev > 0.0
        assert std_dev < 1.0

    def test_per_metric_std_dev_rounded(self):
        evaluator = Evaluator(metrics=["faithfulness"])
        results = _make_results()
        summary = evaluator.summarize(results)

        std_dev = summary["metric_scores"]["faithfulness"]["std_dev"]
        # Should be rounded to 4 decimal places
        assert std_dev == round(std_dev, 4)

    def test_single_sample_metric_stats(self):
        """With one sample, all distribution stats should equal that score."""
        evaluator = Evaluator(metrics=["faithfulness"])
        results = [
            EvalResult(sample_index=0, metrics=[
                MetricResult(name="faithfulness", score=0.85),
            ]),
        ]
        summary = evaluator.summarize(results)

        scores = summary["metric_scores"]["faithfulness"]
        assert scores["median"] == 0.85
        assert scores["p25"] == 0.85
        assert scores["p75"] == 0.85
        assert scores["std_dev"] == 0.0
