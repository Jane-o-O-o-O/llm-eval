"""Tests for the answer correctness metric."""

from unittest.mock import AsyncMock, patch

import pytest

from llm_eval.metrics.answer_correctness import (
    AnswerCorrectnessMetric,
    token_overlap_score,
)
from llm_eval.models import Sample


class TestTokenOverlap:
    """Tests for the token overlap utility."""

    def test_identical_texts(self) -> None:
        score = token_overlap_score("hello world", "hello world")
        assert score == pytest.approx(1.0)

    def test_completely_different(self) -> None:
        score = token_overlap_score("foo bar", "baz qux")
        assert score == pytest.approx(0.0)

    def test_partial_overlap(self) -> None:
        score = token_overlap_score("the cat sat", "the dog sat")
        # common: {the, sat}, answer: {the, cat, sat}, ref: {the, dog, sat}
        # precision = 2/3, recall = 2/3, f1 = 2/3
        assert score == pytest.approx(2 / 3, abs=0.01)

    def test_empty_both(self) -> None:
        assert token_overlap_score("", "") == pytest.approx(1.0)

    def test_empty_answer(self) -> None:
        assert token_overlap_score("", "some text") == pytest.approx(0.0)

    def test_empty_reference(self) -> None:
        assert token_overlap_score("some text", "") == pytest.approx(0.0)

    def test_case_insensitive(self) -> None:
        score = token_overlap_score("Hello WORLD", "hello world")
        assert score == pytest.approx(1.0)

    def test_punctuation_ignored(self) -> None:
        score = token_overlap_score("Hello, world!", "hello world")
        assert score == pytest.approx(1.0)


class TestAnswerCorrectnessMetric:
    """Tests for the AnswerCorrectnessMetric."""

    def test_metric_metadata(self) -> None:
        metric = AnswerCorrectnessMetric()
        assert metric.name == "answer_correctness"
        assert metric.description != ""

    def test_metric_implements_base(self) -> None:
        from llm_eval.metrics import Metric

        metric = AnswerCorrectnessMetric()
        assert isinstance(metric, Metric)

    @pytest.mark.asyncio
    async def test_with_reference_blends_scores(self) -> None:
        metric = AnswerCorrectnessMetric(token_weight=0.4, judge_weight=0.6)
        sample = Sample(
            query="What is 2+2?",
            context=[],
            answer="2+2 equals 4",
            reference="2+2 is 4",
        )

        mock_judge = {"score": 0.9, "reasoning": "Correct answer."}
        with patch.object(metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge):
            result = await metric.evaluate(sample)

        assert result.name == "answer_correctness"
        assert 0.0 <= result.score <= 1.0
        assert result.details["token_overlap"] is not None
        assert result.details["judge_score"] == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_without_reference_uses_judge_only(self) -> None:
        metric = AnswerCorrectnessMetric()
        sample = Sample(
            query="What is Python?",
            context=["Python is a programming language."],
            answer="Python is a programming language.",
            reference=None,
        )

        mock_judge = {"score": 0.85, "reasoning": "Good answer."}
        with patch.object(metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge):
            result = await metric.evaluate(sample)

        assert result.details["token_overlap"] is None
        assert result.score == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_score_clamped_to_1(self) -> None:
        metric = AnswerCorrectnessMetric()
        sample = Sample(
            query="Q",
            context=["ctx"],
            answer="A",
            reference="A",
        )

        mock_judge = {"score": 1.0, "reasoning": "Perfect"}
        with patch.object(metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge):
            result = await metric.evaluate(sample)

        assert result.score <= 1.0

    @pytest.mark.asyncio
    async def test_score_clamped_to_0(self) -> None:
        metric = AnswerCorrectnessMetric()
        sample = Sample(query="Q", context=["ctx"], answer="A")

        mock_judge = {"score": -0.5, "reasoning": "Bad"}
        with patch.object(metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge):
            result = await metric.evaluate(sample)

        assert result.score >= 0.0

    @pytest.mark.asyncio
    async def test_prompt_contains_reference(self) -> None:
        metric = AnswerCorrectnessMetric()
        sample = Sample(
            query="What is X?",
            context=[],
            answer="X is Y",
            reference="X is Y and Z",
        )

        with patch.object(metric, "_judge_call", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"score": 0.8, "reasoning": "OK"}
            await metric.evaluate(sample)
            prompt = mock_call.call_args[0][0]
            assert "X is Y and Z" in prompt

    @pytest.mark.asyncio
    async def test_prompt_without_reference(self) -> None:
        metric = AnswerCorrectnessMetric()
        sample = Sample(query="What?", context=[], answer="That.")

        with patch.object(metric, "_judge_call", new_callable=AsyncMock) as mock_call:
            mock_call.return_value = {"score": 0.5, "reasoning": "OK"}
            await metric.evaluate(sample)
            prompt = mock_call.call_args[0][0]
            assert "Reference Answer" not in prompt

    def test_custom_weights(self) -> None:
        metric = AnswerCorrectnessMetric(token_weight=0.7, judge_weight=0.3)
        assert metric.token_weight == 0.7
        assert metric.judge_weight == 0.3

    @pytest.mark.asyncio
    async def test_weighted_blend_calculation(self) -> None:
        metric = AnswerCorrectnessMetric(token_weight=0.5, judge_weight=0.5)
        sample = Sample(
            query="Q",
            context=[],
            answer="the cat sat on the mat",
            reference="the cat sat on the mat",
        )

        mock_judge = {"score": 0.8, "reasoning": "Good"}
        with patch.object(metric, "_judge_call", new_callable=AsyncMock, return_value=mock_judge):
            result = await metric.evaluate(sample)

        # token_overlap = 1.0, judge = 0.8
        # blended = 0.5 * 1.0 + 0.5 * 0.8 = 0.9
        assert result.score == pytest.approx(0.9, abs=0.01)
