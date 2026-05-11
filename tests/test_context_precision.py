"""Tests for the context_precision metric."""

from unittest.mock import AsyncMock, patch

import pytest
from llm_eval.metrics.context_precision import ContextPrecisionMetric
from llm_eval.models import Sample


class TestContextPrecisionMetric:
    """Tests for ContextPrecisionMetric."""

    def test_metric_metadata(self) -> None:
        metric = ContextPrecisionMetric()
        assert metric.name == "context_precision"
        assert metric.description

    def test_metric_implements_base(self) -> None:
        from llm_eval.metrics import Metric
        metric = ContextPrecisionMetric()
        assert isinstance(metric, Metric)

    @pytest.mark.asyncio
    async def test_high_precision(self) -> None:
        metric = ContextPrecisionMetric()
        sample = Sample(
            query="What is the refund policy?",
            context=[
                "Refunds are processed within 5 business days.",
                "Our store sells electronics.",
                "Contact support at support@example.com.",
            ],
            answer="Refunds take 5 business days.",
            reference="Refunds are processed within 5 business days.",
        )
        with patch.object(metric, "_judge_call", new_callable=AsyncMock) as mock_judge:
            mock_judge.return_value = {"score": 0.67, "reasoning": "2 of 3 chunks relevant"}
            result = await metric.evaluate(sample)
            assert result.score == pytest.approx(0.67)
            assert result.name == "context_precision"

    @pytest.mark.asyncio
    async def test_empty_context(self) -> None:
        metric = ContextPrecisionMetric()
        sample = Sample(
            query="What is X?",
            context=[],
            answer="X is Y.",
        )
        result = await metric.evaluate(sample)
        assert result.score == 0.0

    def test_prompt_contains_context_chunks(self) -> None:
        sample = Sample(
            query="What is X?",
            context=["Chunk A", "Chunk B"],
            answer="Answer",
        )
        prompt = ContextPrecisionMetric._build_prompt(sample)
        assert "[1] Chunk A" in prompt
        assert "[2] Chunk B" in prompt

    @pytest.mark.asyncio
    async def test_perfect_precision(self) -> None:
        metric = ContextPrecisionMetric()
        sample = Sample(
            query="What is the refund policy?",
            context=["Refunds take 5 business days."],
            answer="5 days for refunds.",
        )
        with patch.object(metric, "_judge_call", new_callable=AsyncMock) as mock_judge:
            mock_judge.return_value = {"score": 1.0, "reasoning": "All context relevant"}
            result = await metric.evaluate(sample)
            assert result.score == 1.0
