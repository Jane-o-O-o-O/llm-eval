"""Tests for the hallucination metric."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llm_eval.metrics.hallucination import HallucinationMetric
from llm_eval.models import Sample


@pytest.fixture
def metric() -> HallucinationMetric:
    return HallucinationMetric()


@pytest.fixture
def clean_sample() -> Sample:
    """Sample with no hallucination — answer is fully supported by context."""
    return Sample(
        query="What is Python?",
        context=["Python is a programming language created by Guido van Rossum."],
        answer="Python is a programming language created by Guido van Rossum.",
    )


@pytest.fixture
def hallucinated_sample() -> Sample:
    """Sample with hallucinated claims."""
    return Sample(
        query="What is Python?",
        context=["Python is a programming language."],
        answer="Python is a programming language created by James Gosling in 1995.",
    )


class TestHallucinationMetric:
    def test_name_and_description(self, metric: HallucinationMetric) -> None:
        assert metric.name == "hallucination"
        assert (
            "fabricated" in metric.description.lower()
            or "hallucination" in metric.description.lower()
        )

    @pytest.mark.asyncio
    async def test_clean_answer_scores_low(self, metric, clean_sample) -> None:
        mock_response = {
            "score": 0.0,
            "hallucinated_claims": [],
            "reasoning": "All claims supported",
        }
        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await metric.evaluate(clean_sample)
        assert result.name == "hallucination"
        assert result.score == 0.0
        assert result.details["hallucinated_claims"] == []

    @pytest.mark.asyncio
    async def test_hallucinated_answer_scores_high(self, metric, hallucinated_sample) -> None:
        mock_response = {
            "score": 0.8,
            "hallucinated_claims": ["James Gosling", "1995"],
            "reasoning": "Incorrect creator and year",
        }
        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await metric.evaluate(hallucinated_sample)
        assert result.score == 0.8
        assert "James Gosling" in result.details["hallucinated_claims"]

    @pytest.mark.asyncio
    async def test_score_clamped_to_valid_range(self, metric, clean_sample) -> None:
        mock_response = {"score": 1.5, "hallucinated_claims": [], "reasoning": ""}
        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await metric.evaluate(clean_sample)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_score_clamped_below_zero(self, metric, clean_sample) -> None:
        mock_response = {"score": -0.5, "hallucinated_claims": [], "reasoning": ""}
        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await metric.evaluate(clean_sample)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_prompt_includes_context_and_answer(self, metric, hallucinated_sample) -> None:
        prompt = HallucinationMetric._build_prompt(hallucinated_sample)
        assert "Python is a programming language." in prompt
        assert "James Gosling" in prompt

    @pytest.mark.asyncio
    async def test_default_score_on_empty_response(self, metric, clean_sample) -> None:
        mock_response = {}
        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await metric.evaluate(clean_sample)
        assert result.score == 0.0
