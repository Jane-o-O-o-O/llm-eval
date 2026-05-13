"""Tests for the context_recall metric."""

from unittest.mock import AsyncMock, patch

import pytest

from llm_eval.metrics.context_recall import ContextRecallMetric
from llm_eval.models import Sample


class TestContextRecallMetric:
    """Tests for ContextRecallMetric."""

    def test_metric_metadata(self) -> None:
        metric = ContextRecallMetric()
        assert metric.name == "context_recall"
        assert metric.description

    def test_metric_implements_base(self) -> None:
        from llm_eval.metrics import Metric

        metric = ContextRecallMetric()
        assert isinstance(metric, Metric)

    @pytest.mark.asyncio
    async def test_high_recall(self) -> None:
        metric = ContextRecallMetric()
        sample = Sample(
            query="What is the refund policy?",
            context=[
                "Refunds are processed within 5 business days.",
                "You need a receipt for refunds.",
            ],
            answer="Refunds take 5 days with receipt.",
            reference="Refunds are processed within 5 business days. A receipt is required.",
        )
        with patch.object(metric, "_judge_call", new_callable=AsyncMock) as mock_judge:
            mock_judge.return_value = {"score": 1.0, "reasoning": "All claims covered"}
            result = await metric.evaluate(sample)
            assert result.score == 1.0
            assert result.name == "context_recall"

    @pytest.mark.asyncio
    async def test_no_reference_returns_zero(self) -> None:
        metric = ContextRecallMetric()
        sample = Sample(
            query="What is X?",
            context=["Some context"],
            answer="X is Y.",
            reference=None,
        )
        result = await metric.evaluate(sample)
        assert result.score == 0.0
        assert "Reference" in result.details["reasoning"]

    @pytest.mark.asyncio
    async def test_empty_context_returns_zero(self) -> None:
        metric = ContextRecallMetric()
        sample = Sample(
            query="What is X?",
            context=[],
            answer="X is Y.",
            reference="X is Y because Z.",
        )
        result = await metric.evaluate(sample)
        assert result.score == 0.0

    def test_prompt_contains_reference(self) -> None:
        sample = Sample(
            query="What is X?",
            context=["Context A"],
            answer="Answer",
            reference="Reference answer here",
        )
        prompt = ContextRecallMetric._build_prompt(sample)
        assert "Reference answer here" in prompt
        assert "Context A" in prompt

    @pytest.mark.asyncio
    async def test_low_recall(self) -> None:
        metric = ContextRecallMetric()
        sample = Sample(
            query="What is the refund policy?",
            context=["Our store sells electronics."],
            answer="Refunds take 5 days.",
            reference="Refunds are processed within 5 business days with receipt.",
        )
        with patch.object(metric, "_judge_call", new_callable=AsyncMock) as mock_judge:
            mock_judge.return_value = {"score": 0.0, "reasoning": "Context has no refund info"}
            result = await metric.evaluate(sample)
            assert result.score == 0.0
