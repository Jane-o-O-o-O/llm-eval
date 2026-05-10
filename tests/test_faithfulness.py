"""Tests for the faithfulness metric."""

from unittest.mock import AsyncMock, patch

import pytest
from llm_eval.metrics.faithfulness import FaithfulnessMetric
from llm_eval.models import Sample


class TestFaithfulnessMetric:
    """Tests for the faithfulness RAG metric."""

    def test_metric_metadata(self) -> None:
        metric = FaithfulnessMetric()
        assert metric.name == "faithfulness"
        assert metric.description != ""

    def test_metric_implements_base(self) -> None:
        from llm_eval.metrics import Metric

        metric = FaithfulnessMetric()
        assert isinstance(metric, Metric)

    @pytest.mark.asyncio
    async def test_faithful_answer_scores_high(self) -> None:
        metric = FaithfulnessMetric()
        sample = Sample(
            query="What is the refund policy?",
            context=["Refunds are processed within 5 business days."],
            answer="Refunds are processed within 5 business days.",
        )

        mock_judge_response = {"score": 0.95, "reasoning": "Answer is fully supported by context."}

        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge_response
        ):
            result = await metric.evaluate(sample)
            assert result.name == "faithfulness"
            assert result.score == 0.95
            assert "reasoning" in result.details

    @pytest.mark.asyncio
    async def test_unfaithful_answer_scores_low(self) -> None:
        metric = FaithfulnessMetric()
        sample = Sample(
            query="What is Python?",
            context=["Python is a programming language."],
            answer="Python is a type of snake.",
        )

        mock_judge_response = {
            "score": 0.1,
            "reasoning": "Answer contradicts the context.",
        }

        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge_response
        ):
            result = await metric.evaluate(sample)
            assert result.score == 0.1

    @pytest.mark.asyncio
    async def test_faithfulness_prompt_contains_context_and_answer(self) -> None:
        """Verify the prompt passed to the judge includes key fields."""
        metric = FaithfulnessMetric()
        sample = Sample(
            query="What is X?",
            context=["X is Y.", "X also Z."],
            answer="X is Y and Z.",
        )

        with patch.object(metric, "_judge_call", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"score": 1.0, "reasoning": "OK"}
            await metric.evaluate(sample)
            call_args = mock_call.call_args
            prompt = call_args[0][0]
            assert "X is Y." in prompt
            assert "X is Y and Z." in prompt
            assert "What is X?" in prompt

    @pytest.mark.asyncio
    async def test_faithfulness_with_multiple_context_chunks(self) -> None:
        metric = FaithfulnessMetric()
        sample = Sample(
            query="Tell me about A and B.",
            context=["A is great.", "B is wonderful.", "C is irrelevant."],
            answer="A is great and B is wonderful.",
        )

        mock_judge_response = {"score": 0.9, "reasoning": "Supported by context."}

        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge_response
        ):
            result = await metric.evaluate(sample)
            assert result.score == 0.9
