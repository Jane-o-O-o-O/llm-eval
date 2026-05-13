"""Tests for the format_compliance metric."""

import json

import pytest

from llm_eval.metrics.format_compliance import FormatComplianceMetric
from llm_eval.models import Sample


class TestFormatComplianceMetric:
    """Tests for FormatComplianceMetric (deterministic, no LLM)."""

    def test_metric_metadata(self) -> None:
        metric = FormatComplianceMetric()
        assert metric.name == "format_compliance"
        assert metric.description

    def test_metric_implements_base(self) -> None:
        from llm_eval.metrics import Metric

        metric = FormatComplianceMetric()
        assert isinstance(metric, Metric)

    @pytest.mark.asyncio
    async def test_no_constraints_returns_pass(self) -> None:
        metric = FormatComplianceMetric()
        sample = Sample(
            query="Summarize this.",
            context=["Some context"],
            answer="This is a summary.",
        )
        result = await metric.evaluate(sample)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_valid_json(self) -> None:
        metric = FormatComplianceMetric()
        sample = Sample(
            query="Return JSON.",
            context=["Some context"],
            answer=json.dumps({"key": "value"}),
            metadata={"expected_format": "json"},
        )
        result = await metric.evaluate(sample)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_invalid_json(self) -> None:
        metric = FormatComplianceMetric()
        sample = Sample(
            query="Return JSON.",
            context=["Some context"],
            answer="not json at all",
            metadata={"expected_format": "json"},
        )
        result = await metric.evaluate(sample)
        assert result.score == 0.0
        assert "Invalid JSON" in result.details["checks"][0]["detail"]

    @pytest.mark.asyncio
    async def test_markdown_heading_pass(self) -> None:
        metric = FormatComplianceMetric()
        sample = Sample(
            query="Write a heading.",
            context=["Some context"],
            answer="# My Heading\nSome content",
            metadata={"expected_format": "markdown_heading"},
        )
        result = await metric.evaluate(sample)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_markdown_heading_fail(self) -> None:
        metric = FormatComplianceMetric()
        sample = Sample(
            query="Write a heading.",
            context=["Some context"],
            answer="No heading here",
            metadata={"expected_format": "markdown_heading"},
        )
        result = await metric.evaluate(sample)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_bullet_list_pass(self) -> None:
        metric = FormatComplianceMetric()
        sample = Sample(
            query="List items.",
            context=["Some context"],
            answer="- Item 1\n- Item 2\n- Item 3",
            metadata={"expected_format": "bullet_list"},
        )
        result = await metric.evaluate(sample)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_bullet_list_fail_single_item(self) -> None:
        metric = FormatComplianceMetric()
        sample = Sample(
            query="List items.",
            context=["Some context"],
            answer="- Only one item",
            metadata={"expected_format": "bullet_list"},
        )
        result = await metric.evaluate(sample)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_numbered_list_pass(self) -> None:
        metric = FormatComplianceMetric()
        sample = Sample(
            query="Steps.",
            context=["Some context"],
            answer="1. First step\n2. Second step\n3. Third step",
            metadata={"expected_format": "numbered_list"},
        )
        result = await metric.evaluate(sample)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_word_count_within_bounds(self) -> None:
        metric = FormatComplianceMetric()
        sample = Sample(
            query="Summarize.",
            context=["Some context"],
            answer="A brief summary of the topic.",
            metadata={"min_words": 3, "max_words": 50},
        )
        result = await metric.evaluate(sample)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_word_count_too_short(self) -> None:
        metric = FormatComplianceMetric()
        sample = Sample(
            query="Explain.",
            context=["Some context"],
            answer="Too short",
            metadata={"min_words": 10},
        )
        result = await metric.evaluate(sample)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_word_count_too_long(self) -> None:
        metric = FormatComplianceMetric()
        sample = Sample(
            query="Explain.",
            context=["Some context"],
            answer=" ".join(["word"] * 100),
            metadata={"max_words": 10},
        )
        result = await metric.evaluate(sample)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_unknown_format(self) -> None:
        metric = FormatComplianceMetric()
        sample = Sample(
            query="Do something.",
            context=["Some context"],
            answer="Result",
            metadata={"expected_format": "yaml"},
        )
        result = await metric.evaluate(sample)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_combined_format_and_word_count(self) -> None:
        metric = FormatComplianceMetric()
        sample = Sample(
            query="Return JSON with short answer.",
            context=["Some context"],
            answer=json.dumps({"answer": "yes"}),
            metadata={"expected_format": "json", "max_words": 10},
        )
        result = await metric.evaluate(sample)
        assert result.score == 1.0
