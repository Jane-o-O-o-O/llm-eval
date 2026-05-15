"""Tests for SDK improvements: metadata and metric_options."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llm_eval.sdk import EvalOutput, evaluate, evaluate_file
from llm_eval.models import EvalResult, MetricResult


@pytest.fixture
def mock_evaluator():
    """Mock evaluator for SDK tests."""
    with patch("llm_eval.sdk.Evaluator") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.evaluate = AsyncMock(return_value=[
            EvalResult(sample_index=0, metrics=[
                MetricResult(name="faithfulness", score=0.9),
            ]),
        ])
        mock_instance.summarize.return_value = {
            "total_samples": 1,
            "overall_score": 0.9,
            "pass_count": 1,
            "fail_count": 0,
            "pass_rate": 1.0,
            "threshold": 0.7,
            "metric_scores": {
                "faithfulness": {"mean": 0.9, "min": 0.9, "max": 0.9},
            },
            "median": 0.9,
            "p25": 0.9,
            "p75": 0.9,
            "std_dev": 0.0,
            "min_score": 0.9,
            "max_score": 0.9,
        }
        yield mock_instance


class TestEvalOutputMetadata:
    """Test that EvalOutput includes metadata."""

    @pytest.mark.asyncio
    async def test_evaluate_returns_metadata(self, mock_evaluator):
        output = await evaluate(
            samples=[{"query": "q", "context": ["c"], "answer": "a"}],
            metrics=["faithfulness"],
        )
        assert hasattr(output, "metadata")
        assert isinstance(output.metadata, dict)

    @pytest.mark.asyncio
    async def test_metadata_has_timestamp(self, mock_evaluator):
        output = await evaluate(
            samples=[{"query": "q", "context": ["c"], "answer": "a"}],
            metrics=["faithfulness"],
        )
        assert "timestamp" in output.metadata

    @pytest.mark.asyncio
    async def test_metadata_has_version(self, mock_evaluator):
        output = await evaluate(
            samples=[{"query": "q", "context": ["c"], "answer": "a"}],
            metrics=["faithfulness"],
        )
        assert "version" in output.metadata


class TestSDKMetricOptions:
    """Test that SDK supports metric_options."""

    @pytest.mark.asyncio
    async def test_evaluate_accepts_metric_options(self, mock_evaluator):
        """evaluate() should accept metric_options parameter."""
        output = await evaluate(
            samples=[{"query": "q", "context": ["c"], "answer": "a"}],
            metrics=["faithfulness"],
            metric_options={"faithfulness": {"custom_prompt": "test"}},
        )
        assert isinstance(output, EvalOutput)

    @pytest.mark.asyncio
    async def test_metric_options_passed_to_evaluator(self, mock_evaluator):
        """metric_options should be forwarded to Evaluator."""
        from llm_eval.sdk import Evaluator as SDKEvaluator
        opts = {"faithfulness": {"custom_prompt": "test"}}
        with patch("llm_eval.sdk.Evaluator") as mock_cls:
            mock_cls.return_value = mock_evaluator
            await evaluate(
                samples=[{"query": "q", "context": ["c"], "answer": "a"}],
                metrics=["faithfulness"],
                metric_options=opts,
            )
            mock_cls.assert_called_once()
            _, kwargs = mock_cls.call_args
            assert kwargs.get("metric_options") == opts


class TestEvalOutputProperties:
    """Test EvalOutput properties."""

    def test_passed_true_when_all_pass(self):
        output = EvalOutput(summary={
            "total_samples": 3, "fail_count": 0,
        })
        assert output.passed is True

    def test_passed_false_when_any_fail(self):
        output = EvalOutput(summary={
            "total_samples": 3, "fail_count": 1,
        })
        assert output.passed is False

    def test_passed_false_when_empty(self):
        output = EvalOutput(summary={
            "total_samples": 0, "fail_count": 0,
        })
        assert output.passed is False
