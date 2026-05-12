"""Tests for the toxicity metric."""

from unittest.mock import AsyncMock, patch

import pytest
from llm_eval.metrics.toxicity import ToxicityMetric
from llm_eval.models import Sample


class TestToxicityMetric:
    """Tests for the ToxicityMetric."""

    def test_metric_metadata(self) -> None:
        metric = ToxicityMetric()
        assert metric.name == "toxicity"
        assert metric.description

    def test_metric_implements_base(self) -> None:
        from llm_eval.metrics import Metric
        metric = ToxicityMetric()
        assert isinstance(metric, Metric)

    @pytest.mark.asyncio
    async def test_non_toxic_answer_scores_low(self) -> None:
        """A safe, helpful answer should get a low toxicity score (close to 0.0)."""
        metric = ToxicityMetric()
        sample = Sample(
            query="What is Python?",
            context=["Python is a programming language."],
            answer="Python is a popular programming language used for web development and data science.",
        )
        mock_response = {"score": 0.05, "reasoning": "The answer is safe and professional."}
        with patch.object(metric, "_judge_call", new_callable=AsyncMock, return_value=mock_response):
            result = await metric.evaluate(sample)
            assert result.score == pytest.approx(0.05)
            assert result.name == "toxicity"

    @pytest.mark.asyncio
    async def test_toxic_answer_scores_high(self) -> None:
        """A toxic answer should get a high toxicity score (close to 1.0)."""
        metric = ToxicityMetric()
        sample = Sample(
            query="How do I use this?",
            context=["Read the manual."],
            answer="You are too stupid to understand this.",
        )
        mock_response = {"score": 0.95, "reasoning": "The answer contains an insult directed at the user."}
        with patch.object(metric, "_judge_call", new_callable=AsyncMock, return_value=mock_response):
            result = await metric.evaluate(sample)
            assert result.score == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_score_clamped_to_range(self) -> None:
        """Scores should be clamped to 0.0–1.0."""
        metric = ToxicityMetric()
        sample = Sample(query="q", context=["c"], answer="a")

        # Test upper clamp
        with patch.object(metric, "_judge_call", new_callable=AsyncMock, return_value={"score": 1.5, "reasoning": ""}):
            result = await metric.evaluate(sample)
            assert result.score == 1.0

        # Test lower clamp
        with patch.object(metric, "_judge_call", new_callable=AsyncMock, return_value={"score": -0.1, "reasoning": ""}):
            result = await metric.evaluate(sample)
            assert result.score == 0.0

    def test_prompt_contains_query_and_answer(self) -> None:
        sample = Sample(
            query="Is this safe?",
            context=["Context here."],
            answer="Everything is fine.",
        )
        prompt = ToxicityMetric._build_prompt(sample)
        assert "Is this safe?" in prompt
        assert "Everything is fine." in prompt

    def test_prompt_includes_toxicity_categories(self) -> None:
        """The prompt should mention toxicity evaluation categories."""
        sample = Sample(query="q", context=["c"], answer="a")
        prompt = ToxicityMetric._build_prompt(sample)
        # Should reference common toxicity aspects
        assert "toxic" in prompt.lower() or "harmful" in prompt.lower() or "offensive" in prompt.lower()
