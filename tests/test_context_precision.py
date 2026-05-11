"""Tests for the context_precision metric."""

from unittest.mock import AsyncMock, patch

import pytest
from llm_eval.metrics.context_precision import ContextPrecisionMetric
from llm_eval.models import Sample


class TestContextPrecisionMetric:
    """Tests for the context_precision RAG metric."""

    def test_metric_metadata(self) -> None:
        metric = ContextPrecisionMetric()
        assert metric.name == "context_precision"
        assert metric.description != ""

    def test_metric_implements_base(self) -> None:
        from llm_eval.metrics import Metric

        metric = ContextPrecisionMetric()
        assert isinstance(metric, Metric)

    @pytest.mark.asyncio
    async def test_high_precision_when_relevant_chunks_ranked_first(self) -> None:
        """When relevant context is at the top, precision should be high."""
        metric = ContextPrecisionMetric()
        sample = Sample(
            query="What is the refund policy?",
            context=[
                "Refunds are processed within 5 business days.",
                "Our company was founded in 2010.",
                "Refunds require a valid receipt.",
            ],
            answer="Refunds are processed within 5 business days and require a valid receipt.",
            reference="Refunds are processed within 5 business days and require a valid receipt.",
        )

        # Judge says chunks 1 and 3 are relevant (positions 0 and 2 in 0-indexed)
        mock_judge_response = {
            "score": 0.83,
            "reasoning": "Two relevant chunks ranked at positions 1 and 3.",
            "relevant_positions": [1, 3],
        }

        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge_response
        ):
            result = await metric.evaluate(sample)
            assert result.name == "context_precision"
            assert 0.0 <= result.score <= 1.0
            assert result.score == pytest.approx(0.83)
            assert "reasoning" in result.details

    @pytest.mark.asyncio
    async def test_low_precision_when_relevant_chunks_ranked_last(self) -> None:
        """When relevant context is at the bottom, precision should be low."""
        metric = ContextPrecisionMetric()
        sample = Sample(
            query="What is the refund policy?",
            context=[
                "Our company was founded in 2010.",
                "We sell electronics.",
                "Refunds are processed within 5 business days.",
            ],
            answer="Refunds are processed within 5 business days.",
            reference="Refunds are processed within 5 business days.",
        )

        mock_judge_response = {
            "score": 0.33,
            "reasoning": "Relevant chunk only at last position.",
            "relevant_positions": [3],
        }

        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge_response
        ):
            result = await metric.evaluate(sample)
            assert result.score == pytest.approx(0.33)

    @pytest.mark.asyncio
    async def test_perfect_precision(self) -> None:
        """All context chunks are relevant — perfect precision."""
        metric = ContextPrecisionMetric()
        sample = Sample(
            query="What is Python?",
            context=[
                "Python is a programming language.",
                "Python was created by Guido van Rossum.",
            ],
            answer="Python is a programming language created by Guido van Rossum.",
            reference="Python is a programming language created by Guido van Rossum.",
        )

        mock_judge_response = {
            "score": 1.0,
            "reasoning": "All chunks are relevant and ranked first.",
            "relevant_positions": [1, 2],
        }

        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge_response
        ):
            result = await metric.evaluate(sample)
            assert result.score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_prompt_contains_context_chunks_numbered(self) -> None:
        """Verify the prompt passes numbered context chunks to the judge."""
        metric = ContextPrecisionMetric()
        sample = Sample(
            query="What is X?",
            context=["X is Y.", "X also Z.", "W is unrelated."],
            answer="X is Y and Z.",
            reference="X is Y and Z.",
        )

        with patch.object(metric, "_judge_call", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"score": 0.67, "reasoning": "OK", "relevant_positions": [1, 2]}
            await metric.evaluate(sample)
            prompt = mock_call.call_args[0][0]
            assert "What is X?" in prompt
            assert "X is Y." in prompt
            assert "X also Z." in prompt
