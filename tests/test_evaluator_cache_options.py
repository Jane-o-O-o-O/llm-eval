"""Tests for metric options and evaluator cache integration."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_eval.evaluator import Evaluator
from llm_eval.metrics import Metric, MetricRegistry, MetricResult
from llm_eval.models import EvalResult, JudgeConfig, Sample


class TestMetricOptions:
    """Test metric_options support in metrics."""

    def test_metric_accepts_options(self):
        """Metric base class should accept metric_options."""
        # Create a minimal concrete metric for testing
        class TestMetric(Metric):
            name = "test"
            description = "test metric"

            async def evaluate(self, sample: Sample) -> MetricResult:
                return MetricResult(name=self.name, score=1.0)

        metric = TestMetric(metric_options={"custom_prompt": "custom"})
        assert metric._metric_options == {"custom_prompt": "custom"}

    def test_metric_default_empty_options(self):
        """Metric should default to empty options."""
        class TestMetric(Metric):
            name = "test"
            description = "test metric"

            async def evaluate(self, sample: Sample) -> MetricResult:
                return MetricResult(name=self.name, score=1.0)

        metric = TestMetric()
        assert metric._metric_options == {}

    def test_metric_accepts_cache_params(self):
        """Metric should accept cache and use_cache parameters."""
        class TestMetric(Metric):
            name = "test"
            description = "test metric"

            async def evaluate(self, sample: Sample) -> MetricResult:
                return MetricResult(name=self.name, score=1.0)

        mock_cache = MagicMock()
        metric = TestMetric(cache=mock_cache, use_cache=False)
        assert metric._cache is mock_cache
        assert metric._use_cache is False


class TestEvaluatorCacheIntegration:
    """Test that Evaluator properly passes cache to metrics."""

    def test_evaluator_use_cache_default(self):
        """Evaluator should default to use_cache=True."""
        evaluator = Evaluator(metrics=["faithfulness"])
        assert evaluator.use_cache is True

    def test_evaluator_use_cache_false(self):
        """Evaluator should accept use_cache=False."""
        evaluator = Evaluator(metrics=["faithfulness"], use_cache=False)
        assert evaluator.use_cache is False

    def test_evaluator_metric_options(self):
        """Evaluator should store metric_options."""
        options = {"faithfulness": {"custom_prompt": "test"}}
        evaluator = Evaluator(
            metrics=["faithfulness"],
            metric_options=options,
        )
        assert evaluator.metric_options == options

    def test_evaluator_re_registers_metrics_with_cache(self):
        """Evaluator should re-register metrics with cache when use_cache=True."""
        evaluator = Evaluator(
            metrics=["faithfulness"],
            use_cache=True,
        )
        # The registry should have metrics
        assert evaluator._registry.is_registered("faithfulness")
        metric = evaluator._registry.get("faithfulness")
        # Metric should have cache-related attributes
        assert hasattr(metric, "_cache")
        assert hasattr(metric, "_use_cache")


class TestEvaluatorSummarize:
    """Test evaluator summarize with various inputs."""

    def test_summarize_empty_results(self):
        """Summarize should handle empty results gracefully."""
        evaluator = Evaluator(metrics=["faithfulness"])
        summary = evaluator.summarize([])
        assert summary["total_samples"] == 0
        assert summary["overall_score"] == 0.0

    def test_summarize_single_result(self):
        """Summarize should compute correct stats for single result."""
        evaluator = Evaluator(metrics=["faithfulness"])
        result = EvalResult(
            sample_index=0,
            metrics=[MetricResult(name="faithfulness", score=0.8)],
        )
        summary = evaluator.summarize([result])
        assert summary["total_samples"] == 1
        assert summary["overall_score"] == 0.8
        assert summary["pass_count"] == 1
        assert summary["fail_count"] == 0

    def test_summarize_with_weights(self):
        """Summarize should use metric_weights for overall score."""
        evaluator = Evaluator(
            metrics=["faithfulness", "answer_relevancy"],
            metric_weights={"faithfulness": 2.0, "answer_relevancy": 1.0},
        )
        results = [
            EvalResult(
                sample_index=0,
                metrics=[
                    MetricResult(name="faithfulness", score=0.9),
                    MetricResult(name="answer_relevancy", score=0.6),
                ],
            ),
        ]
        summary = evaluator.summarize(results)
        # Weighted: (0.9*2.0 + 0.6*1.0) / (2.0+1.0) = 2.4/3.0 = 0.8
        assert abs(summary["overall_score"] - 0.8) < 0.001
