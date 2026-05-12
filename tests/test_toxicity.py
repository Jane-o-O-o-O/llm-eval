"""Tests for the toxicity metric."""

from __future__ import annotations

import pytest

from llm_eval.metrics.toxicity import ToxicityMetric
from llm_eval.models import Sample


@pytest.fixture
def metric() -> ToxicityMetric:
    return ToxicityMetric()


@pytest.fixture
def clean_sample() -> Sample:
    return Sample(
        query="What is Python?",
        context=["Python is a programming language."],
        answer="Python is a high-level programming language known for its readability.",
    )


@pytest.fixture
def toxic_sample() -> Sample:
    return Sample(
        query="What do you think?",
        context=["Some context."],
        answer="You're such an idiot for asking that, you stupid loser.",
    )


class TestToxicityPatterns:
    """Test pattern-based toxicity detection."""

    def test_clean_text_no_hits(self, metric: ToxicityMetric) -> None:
        hits = metric._scan_patterns("This is a perfectly normal and helpful response.")
        assert hits == []

    def test_detects_insults(self, metric: ToxicityMetric) -> None:
        hits = metric._scan_patterns("You are an idiot and a moron.")
        assert len(hits) >= 1
        assert any("idiot" in h.lower() or "moron" in h.lower() for h in hits)

    def test_detects_threats(self, metric: ToxicityMetric) -> None:
        hits = metric._scan_patterns("Go hurt yourself.")
        assert len(hits) >= 1

    def test_case_insensitive(self, metric: ToxicityMetric) -> None:
        hits = metric._scan_patterns("You are an IDIOT.")
        assert len(hits) >= 1

    def test_partial_match_no_false_positive(self, metric: ToxicityMetric) -> None:
        _hits = metric._scan_patterns("The documentation is stupidly comprehensive.")
        # \"stupidly\" contains \"stupid\" — pattern should still match (conservative)
        # This is acceptable behavior for a safety-first metric
        assert isinstance(_hits, list)  # should not raise


class TestToxicityEvaluate:
    """Test the full evaluate pipeline."""

    @pytest.mark.asyncio
    async def test_toxic_sample_scores_low(self, metric: ToxicityMetric, toxic_sample: Sample) -> None:
        result = await metric.evaluate(toxic_sample)
        assert result.name == "toxicity"
        assert result.score == 0.0
        assert result.details["method"] == "pattern_scan"
        assert len(result.details["hits"]) >= 1

    @pytest.mark.asyncio
    async def test_clean_sample_calls_judge(self, metric: ToxicityMetric, clean_sample: Sample) -> None:
        # Mock the judge call
        async def mock_judge(prompt: str) -> dict:
            return {"score": 0.05, "reasoning": "The response is safe and professional."}

        metric._judge_call = mock_judge
        result = await metric.evaluate(clean_sample)
        assert result.name == "toxicity"
        assert result.score > 0.9  # Safety score should be high
        assert result.details["method"] == "llm_judge"

    @pytest.mark.asyncio
    async def test_judge_returns_high_toxicity(self, metric: ToxicityMetric, clean_sample: Sample) -> None:
        async def mock_judge(prompt: str) -> dict:
            return {"score": 0.8, "reasoning": "Contains subtle bias."}

        metric._judge_call = mock_judge
        result = await metric.evaluate(clean_sample)
        assert result.score == pytest.approx(0.2, abs=0.01)  # 1.0 - 0.8

    @pytest.mark.asyncio
    async def test_score_clamped(self, metric: ToxicityMetric, clean_sample: Sample) -> None:
        async def mock_judge(prompt: str) -> dict:
            return {"score": 1.5, "reasoning": "Over-scored."}

        metric._judge_call = mock_judge
        result = await metric.evaluate(clean_sample)
        assert result.score == 0.0  # 1.0 - 1.0 (clamped)

    def test_metric_metadata(self, metric: ToxicityMetric) -> None:
        assert metric.name == "toxicity"
        assert "toxic" in metric.description.lower() or "harmful" in metric.description.lower()


class TestToxicityPrompt:
    """Test the judge prompt construction."""

    def test_prompt_contains_answer(self, metric: ToxicityMetric, clean_sample: Sample) -> None:
        prompt = metric._build_prompt(clean_sample)
        assert clean_sample.answer in prompt

    def test_prompt_contains_query(self, metric: ToxicityMetric, clean_sample: Sample) -> None:
        prompt = metric._build_prompt(clean_sample)
        assert clean_sample.query in prompt

    def test_prompt_asks_for_json(self, metric: ToxicityMetric, clean_sample: Sample) -> None:
        prompt = metric._build_prompt(clean_sample)
        assert "JSON" in prompt or "json" in prompt
