"""Tests for the context_recall metric."""

from unittest.mock import AsyncMock, patch

import pytest
from llm_eval.metrics.context_recall import ContextRecallMetric
from llm_eval.models import Sample


class TestContextRecallMetric:
    """Tests for the context_recall RAG metric."""

    def test_metric_metadata(self) -> None:
        metric = ContextRecallMetric()
        assert metric.name == "context_recall"
        assert metric.description != ""

    def test_metric_implements_base(self) -> None:
        from llm_eval.metrics import Metric

        metric = ContextRecallMetric()
        assert isinstance(metric, Metric)

    @pytest.mark.asyncio
    async def test_high_recall_when_context_covers_reference(self) -> None:
        """When context covers all claims in reference, recall should be high."""
        metric = ContextRecallMetric()
        sample = Sample(
            query="What is the refund policy?",
            context=[
                "Refunds are processed within 5 business days.",
                "Refunds require a valid receipt.",
            ],
            answer="Refunds take 5 business days and need a receipt.",
            reference="Refunds are processed within 5 business days and require a valid receipt.",
        )

        mock_judge_response = {
            "score": 1.0,
            "reasoning": "All reference claims are covered by the context.",
            "claims_supported": 2,
            "claims_total": 2,
        }

        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge_response
        ):
            result = await metric.evaluate(sample)
            assert result.name == "context_recall"
            assert result.score == pytest.approx(1.0)
            assert "reasoning" in result.details

    @pytest.mark.asyncio
    async def test_low_recall_when_context_misses_reference_claims(self) -> None:
        """When context misses important claims, recall should be low."""
        metric = ContextRecallMetric()
        sample = Sample(
            query="What is the refund policy?",
            context=["Our company was founded in 2010."],
            answer="No refund info available.",
            reference="Refunds are processed within 5 business days and require a valid receipt.",
        )

        mock_judge_response = {
            "score": 0.0,
            "reasoning": "Context does not contain any refund-related information.",
            "claims_supported": 0,
            "claims_total": 2,
        }

        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge_response
        ):
            result = await metric.evaluate(sample)
            assert result.score == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_partial_recall(self) -> None:
        """When context covers some but not all claims, recall is partial."""
        metric = ContextRecallMetric()
        sample = Sample(
            query="What is the refund policy?",
            context=["Refunds are processed within 5 business days."],
            answer="Refunds take 5 business days.",
            reference="Refunds are processed within 5 business days and require a valid receipt.",
        )

        mock_judge_response = {
            "score": 0.5,
            "reasoning": "Only one of two reference claims is covered by context.",
            "claims_supported": 1,
            "claims_total": 2,
        }

        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge_response
        ):
            result = await metric.evaluate(sample)
            assert result.score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_prompt_contains_reference_and_context(self) -> None:
        """Verify the judge prompt includes reference and context."""
        metric = ContextRecallMetric()
        sample = Sample(
            query="What is X?",
            context=["X is Y.", "X also Z."],
            answer="X is Y and Z.",
            reference="X is Y and also Z.",
        )

        with patch.object(metric, "_judge_call", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"score": 1.0, "reasoning": "OK", "claims_supported": 1, "claims_total": 1}
            await metric.evaluate(sample)
            prompt = mock_call.call_args[0][0]
            assert "What is X?" in prompt
            assert "X is Y and also Z." in prompt
            assert "X is Y." in prompt
