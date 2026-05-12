"""Tests for metric weights in evaluator and config."""

from __future__ import annotations

import pytest

from llm_eval.evaluator import Evaluator
from llm_eval.models import EvalConfig, EvalResult, MetricResult


class TestMetricWeightsConfig:
    """Tests for metric_weights in EvalConfig."""

    def test_default_weights_empty(self) -> None:
        config = EvalConfig.from_dict({"evaluations": []})
        assert config.metric_weights == {}

    def test_weights_from_dict(self) -> None:
        data = {
            "defaults": {
                "metric_weights": {"faithfulness": 2.0, "answer_relevancy": 1.0}
            },
            "evaluations": [],
        }
        config = EvalConfig.from_dict(data)
        assert config.metric_weights == {"faithfulness": 2.0, "answer_relevancy": 1.0}

    def test_weights_missing_defaults(self) -> None:
        config = EvalConfig.from_dict({})
        assert config.metric_weights == {}


class TestEvaluatorWeights:
    """Tests for weighted overall score in Evaluator."""

    def _make_results(self) -> list[EvalResult]:
        """Create sample results with two metrics."""
        return [
            EvalResult(
                sample_index=0,
                metrics=[
                    MetricResult(name="faithfulness", score=0.9),
                    MetricResult(name="answer_relevancy", score=0.6),
                ],
            ),
            EvalResult(
                sample_index=1,
                metrics=[
                    MetricResult(name="faithfulness", score=0.8),
                    MetricResult(name="answer_relevancy", score=0.7),
                ],
            ),
        ]

    def test_equal_weights_same_as_default(self) -> None:
        results = self._make_results()
        ev_default = Evaluator(metrics=["faithfulness", "answer_relevancy"])
        ev_equal = Evaluator(
            metrics=["faithfulness", "answer_relevancy"],
            metric_weights={"faithfulness": 1.0, "answer_relevancy": 1.0},
        )
        s1 = ev_default.summarize(results)
        s2 = ev_equal.summarize(results)
        assert s1["overall_score"] == pytest.approx(s2["overall_score"], abs=0.001)

    def test_weighted_shifts_toward_higher_weight(self) -> None:
        results = self._make_results()
        # faithfulness scores: 0.9, 0.8 → mean 0.85
        # answer_relevancy scores: 0.6, 0.7 → mean 0.65
        ev = Evaluator(
            metrics=["faithfulness", "answer_relevancy"],
            metric_weights={"faithfulness": 3.0, "answer_relevancy": 1.0},
        )
        summary = ev.summarize(results)
        # weighted = (0.85 * 3 + 0.65 * 1) / 4 = (2.55 + 0.65) / 4 = 0.8
        assert summary["overall_score"] == pytest.approx(0.8, abs=0.01)

    def test_weight_default_for_unspecified_metric(self) -> None:
        results = self._make_results()
        ev = Evaluator(
            metrics=["faithfulness", "answer_relevancy"],
            metric_weights={"faithfulness": 2.0},  # answer_relevancy defaults to 1.0
        )
        summary = ev.summarize(results)
        # weighted = (0.85 * 2 + 0.65 * 1) / 3 = (1.7 + 0.65) / 3 = 0.783
        assert summary["overall_score"] == pytest.approx(0.783, abs=0.01)

    def test_empty_weights_uses_equal(self) -> None:
        results = self._make_results()
        ev = Evaluator(metrics=["faithfulness", "answer_relevancy"], metric_weights={})
        summary = ev.summarize(results)
        # Equal weight: (0.85 + 0.65) / 2 = 0.75
        assert summary["overall_score"] == pytest.approx(0.75, abs=0.01)
