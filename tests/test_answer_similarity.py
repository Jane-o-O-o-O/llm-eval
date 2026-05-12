"""Tests for the answer similarity metric."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llm_eval.metrics.answer_similarity import AnswerSimilarityMetric
from llm_eval.models import Sample


@pytest.fixture
def metric() -> AnswerSimilarityMetric:
    return AnswerSimilarityMetric()


@pytest.fixture
def sample_with_ref() -> Sample:
    return Sample(
        query="What is Python?",
        context=["Python is a programming language."],
        answer="Python is a widely-used programming language.",
        reference="Python is a popular programming language.",
    )


@pytest.fixture
def sample_without_ref() -> Sample:
    return Sample(
        query="What is Python?",
        context=["Python is a programming language."],
        answer="Python is a programming language.",
    )


class TestAnswerSimilarityMetric:
    def test_name_and_description(self, metric: AnswerSimilarityMetric) -> None:
        assert metric.name == "answer_similarity"
        assert "similarity" in metric.description.lower()

    @pytest.mark.asyncio
    async def test_high_similarity(self, metric, sample_with_ref) -> None:
        mock_response = {"score": 0.9, "reasoning": "Very similar meaning"}
        with patch.object(metric, "_judge_call", new_callable=AsyncMock, return_value=mock_response):
            result = await metric.evaluate(sample_with_ref)
        assert result.name == "answer_similarity"
        assert result.score == 0.9
        assert "reasoning" in result.details

    @pytest.mark.asyncio
    async def test_no_reference_returns_zero(self, metric, sample_without_ref) -> None:
        result = await metric.evaluate(sample_without_ref)
        assert result.score == 0.0
        assert "warning" in result.details

    @pytest.mark.asyncio
    async def test_score_clamped(self, metric, sample_with_ref) -> None:
        mock_response = {"score": 2.0, "reasoning": ""}
        with patch.object(metric, "_judge_call", new_callable=AsyncMock, return_value=mock_response):
            result = await metric.evaluate(sample_with_ref)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_score_clamped_below_zero(self, metric, sample_with_ref) -> None:
        mock_response = {"score": -1.0, "reasoning": ""}
        with patch.object(metric, "_judge_call", new_callable=AsyncMock, return_value=mock_response):
            result = await metric.evaluate(sample_with_ref)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_prompt_includes_reference(self, metric, sample_with_ref) -> None:
        prompt = AnswerSimilarityMetric._build_prompt(sample_with_ref)
        assert "popular programming language" in prompt
        assert "widely-used" in prompt

    @pytest.mark.asyncio
    async def test_default_score_on_empty_response(self, metric, sample_with_ref) -> None:
        mock_response = {}
        with patch.object(metric, "_judge_call", new_callable=AsyncMock, return_value=mock_response):
            result = await metric.evaluate(sample_with_ref)
        assert result.score == 0.0
