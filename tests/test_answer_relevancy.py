"""Tests for the answer_relevancy metric."""

from unittest.mock import AsyncMock, patch

import pytest

from llm_eval.metrics.answer_relevancy import AnswerRelevancyMetric
from llm_eval.models import Sample


class TestAnswerRelevancyMetric:
    """Tests for the answer_relevancy RAG metric."""

    def test_metric_metadata(self) -> None:
        metric = AnswerRelevancyMetric()
        assert metric.name == "answer_relevancy"
        assert metric.description != ""

    def test_metric_implements_base(self) -> None:
        from llm_eval.metrics import Metric

        metric = AnswerRelevancyMetric()
        assert isinstance(metric, Metric)

    @pytest.mark.asyncio
    async def test_relevant_answer_scores_high(self) -> None:
        metric = AnswerRelevancyMetric()
        sample = Sample(
            query="How do I reset my password?",
            context=["Click 'Forgot Password' on the login page."],
            answer="Navigate to the login page and click 'Forgot Password' to reset your password.",
        )

        mock_judge_response = {
            "score": 0.95,
            "reasoning": "Answer directly addresses the password reset query.",
        }

        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge_response
        ):
            result = await metric.evaluate(sample)
            assert result.name == "answer_relevancy"
            assert result.score == 0.95

    @pytest.mark.asyncio
    async def test_irrelevant_answer_scores_low(self) -> None:
        metric = AnswerRelevancyMetric()
        sample = Sample(
            query="How do I reset my password?",
            context=["Click 'Forgot Password' on the login page."],
            answer="The weather today is sunny with a high of 75°F.",
        )

        mock_judge_response = {
            "score": 0.05,
            "reasoning": "Answer is completely irrelevant to the query.",
        }

        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge_response
        ):
            result = await metric.evaluate(sample)
            assert result.score < 0.2

    @pytest.mark.asyncio
    async def test_relevancy_prompt_contains_query_and_answer(self) -> None:
        metric = AnswerRelevancyMetric()
        sample = Sample(
            query="What is the capital of France?",
            context=["France is in Europe."],
            answer="Paris is the capital of France.",
        )

        with patch.object(metric, "_judge_call", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"score": 1.0, "reasoning": "OK"}
            await metric.evaluate(sample)
            prompt = mock_call.call_args[0][0]
            assert "What is the capital of France?" in prompt
            assert "Paris is the capital of France." in prompt
