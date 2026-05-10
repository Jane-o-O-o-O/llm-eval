"""Tests for the evaluator engine."""

from unittest.mock import AsyncMock, patch

import pytest
from llm_eval.evaluator import Evaluator
from llm_eval.metrics import MetricResult
from llm_eval.models import EvalResult, JudgeConfig, Sample


class TestEvaluator:
    """Tests for the Evaluator engine."""

    def test_create_evaluator(self) -> None:
        evaluator = Evaluator(metrics=["faithfulness"])
        assert evaluator.metric_names == ["faithfulness"]

    def test_create_evaluator_multiple_metrics(self) -> None:
        evaluator = Evaluator(metrics=["faithfulness", "answer_relevancy"])
        assert len(evaluator.metric_names) == 2

    @pytest.mark.asyncio
    async def test_evaluate_single_sample(self) -> None:
        evaluator = Evaluator(metrics=["faithfulness"])

        sample = Sample(
            query="What is Python?",
            context=["Python is a programming language."],
            answer="Python is a programming language.",
        )

        mock_result = MetricResult(name="faithfulness", score=0.9, details={"reasoning": "Good"})

        with patch.object(
            evaluator, "_run_metric", new_callable=AsyncMock, return_value=mock_result
        ):
            result = await evaluator.evaluate_sample(sample, index=0)
            assert isinstance(result, EvalResult)
            assert result.sample_index == 0
            assert len(result.metrics) == 1
            assert result.metrics[0].score == 0.9

    @pytest.mark.asyncio
    async def test_evaluate_multiple_samples(self) -> None:
        evaluator = Evaluator(metrics=["faithfulness"])

        samples = [
            Sample(query="q1", context=["c1"], answer="a1"),
            Sample(query="q2", context=["c2"], answer="a2"),
        ]

        mock_result = MetricResult(name="faithfulness", score=0.8, details={})

        with patch.object(
            evaluator, "_run_metric", new_callable=AsyncMock, return_value=mock_result
        ):
            results = await evaluator.evaluate(samples)
            assert len(results) == 2
            assert results[0].sample_index == 0
            assert results[1].sample_index == 1

    @pytest.mark.asyncio
    async def test_evaluate_with_threshold_pass(self) -> None:
        evaluator = Evaluator(metrics=["faithfulness"], threshold=0.7)

        sample = Sample(query="q", context=["c"], answer="a")
        mock_result = MetricResult(name="faithfulness", score=0.9, details={})

        with patch.object(
            evaluator, "_run_metric", new_callable=AsyncMock, return_value=mock_result
        ):
            result = await evaluator.evaluate_sample(sample, index=0)
            assert result.overall_score >= 0.7

    @pytest.mark.asyncio
    async def test_evaluate_with_threshold_fail(self) -> None:
        evaluator = Evaluator(metrics=["faithfulness"], threshold=0.8)

        sample = Sample(query="q", context=["c"], answer="a")
        mock_result = MetricResult(name="faithfulness", score=0.5, details={})

        with patch.object(
            evaluator, "_run_metric", new_callable=AsyncMock, return_value=mock_result
        ):
            result = await evaluator.evaluate_sample(sample, index=0)
            assert result.overall_score < 0.8

    def test_summary_with_results(self) -> None:
        evaluator = Evaluator(metrics=["faithfulness"], threshold=0.7)
        results = [
            EvalResult(
                sample_index=0,
                metrics=[MetricResult(name="faithfulness", score=0.9, details={})],
            ),
            EvalResult(
                sample_index=1,
                metrics=[MetricResult(name="faithfulness", score=0.5, details={})],
            ),
        ]
        summary = evaluator.summarize(results)
        assert summary["total_samples"] == 2
        assert summary["overall_score"] == pytest.approx(0.7)
        assert summary["pass_count"] == 1
        assert summary["fail_count"] == 1
        assert "faithfulness" in summary["metric_scores"]

    def test_summary_empty_results(self) -> None:
        evaluator = Evaluator(metrics=["faithfulness"])
        summary = evaluator.summarize([])
        assert summary["total_samples"] == 0
        assert summary["overall_score"] == 0.0
