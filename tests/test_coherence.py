"""Tests for the coherence metric."""

from unittest.mock import AsyncMock, patch

import pytest
from llm_eval.metrics.coherence import CoherenceMetric
from llm_eval.models import Sample


class TestCoherenceMetric:
    """Tests for the coherence metric."""

    def test_metric_metadata(self) -> None:
        metric = CoherenceMetric()
        assert metric.name == "coherence"
        assert "quality" in metric.description.lower() or "coherence" in metric.description.lower()

    def test_metric_implements_base(self) -> None:
        from llm_eval.metrics import Metric
        metric = CoherenceMetric()
        assert isinstance(metric, Metric)

    @pytest.mark.asyncio
    async def test_coherent_answer_scores_high(self) -> None:
        metric = CoherenceMetric()
        sample = Sample(
            query="Explain how a CPU works.",
            context=["A CPU processes instructions."],
            answer=(
                "A CPU (Central Processing Unit) is the primary component of a computer. "
                "It fetches instructions from memory, decodes them, and executes them. "
                "The process repeats in a cycle known as the fetch-decode-execute cycle."
            ),
        )

        mock_response = {"score": 0.92, "reasoning": "Well-structured with clear flow."}

        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await metric.evaluate(sample)
            assert result.name == "coherence"
            assert result.score == 0.92
            assert "reasoning" in result.details

    @pytest.mark.asyncio
    async def test_incoherent_answer_scores_low(self) -> None:
        metric = CoherenceMetric()
        sample = Sample(
            query="Explain how a CPU works.",
            context=["A CPU processes instructions."],
            answer="CPU good. Fast. Computer. Works. Instructions.",
        )

        mock_response = {"score": 0.2, "reasoning": "Fragmented, no clear structure."}

        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await metric.evaluate(sample)
            assert result.score == 0.2

    @pytest.mark.asyncio
    async def test_score_clamped_to_valid_range(self) -> None:
        metric = CoherenceMetric()
        sample = Sample(query="q", context=["c"], answer="a")

        # Score above 1.0 should be clamped
        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock,
            return_value={"score": 1.5, "reasoning": "test"}
        ):
            result = await metric.evaluate(sample)
            assert result.score == 1.0

        # Score below 0.0 should be clamped
        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock,
            return_value={"score": -0.5, "reasoning": "test"}
        ):
            result = await metric.evaluate(sample)
            assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_default_score_on_empty_response(self) -> None:
        metric = CoherenceMetric()
        sample = Sample(query="q", context=["c"], answer="a")

        with patch.object(
            metric, "_judge_call", new_callable=AsyncMock, return_value={}
        ):
            result = await metric.evaluate(sample)
            assert result.score == 0.0  # defaults to 0.0

    def test_build_prompt_contains_key_sections(self) -> None:
        sample = Sample(
            query="What is AI?",
            context=["AI is artificial intelligence."],
            answer="AI stands for artificial intelligence.",
        )
        prompt = CoherenceMetric._build_prompt(sample)
        assert "What is AI?" in prompt
        assert "AI stands for artificial intelligence." in prompt
        assert "JSON" in prompt
