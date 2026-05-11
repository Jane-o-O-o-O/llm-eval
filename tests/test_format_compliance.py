"""Tests for the format_compliance metric."""

import pytest
from llm_eval.metrics.format_compliance import FormatComplianceMetric
from llm_eval.models import Sample


class TestFormatComplianceMetric:
    """Tests for the format_compliance metric."""

    def test_metric_metadata(self) -> None:
        metric = FormatComplianceMetric()
        assert metric.name == "format_compliance"
        assert metric.description != ""

    def test_metric_implements_base(self) -> None:
        from llm_eval.metrics import Metric

        metric = FormatComplianceMetric()
        assert isinstance(metric, Metric)

    @pytest.mark.asyncio
    async def test_answer_within_max_length(self) -> None:
        metric = FormatComplianceMetric(max_length=100)
        sample = Sample(
            query="What is Python?",
            context=["Python is a programming language."],
            answer="Python is a programming language.",
        )
        result = await metric.evaluate(sample)
        assert result.name == "format_compliance"
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_answer_exceeds_max_length(self) -> None:
        metric = FormatComplianceMetric(max_length=10)
        sample = Sample(
            query="What is Python?",
            context=["Python is a programming language."],
            answer="Python is a programming language.",
        )
        result = await metric.evaluate(sample)
        assert result.score < 1.0
        assert result.details["length_penalty"] == 1

    @pytest.mark.asyncio
    async def test_answer_with_forbidden_patterns(self) -> None:
        metric = FormatComplianceMetric(forbidden_patterns=["I don't know", "sorry"])
        sample = Sample(
            query="What is Python?",
            context=["Python is a programming language."],
            answer="I don't know what Python is.",
        )
        result = await metric.evaluate(sample)
        assert result.score < 1.0
        assert len(result.details["matched_forbidden"]) > 0

    @pytest.mark.asyncio
    async def test_answer_no_forbidden_patterns(self) -> None:
        metric = FormatComplianceMetric(forbidden_patterns=["I don't know"])
        sample = Sample(
            query="What is Python?",
            context=["Python is a programming language."],
            answer="Python is a programming language.",
        )
        result = await metric.evaluate(sample)
        assert result.score == 1.0
        assert len(result.details["matched_forbidden"]) == 0

    @pytest.mark.asyncio
    async def test_answer_with_required_sections_present(self) -> None:
        metric = FormatComplianceMetric(required_sections=["summary", "details"])
        sample = Sample(
            query="Explain Python.",
            context=["Python is a programming language."],
            answer="Summary: Python is great.\nDetails: It is used for many things.",
        )
        result = await metric.evaluate(sample)
        assert result.score == 1.0
        assert result.details["missing_sections"] == []

    @pytest.mark.asyncio
    async def test_answer_missing_required_sections(self) -> None:
        metric = FormatComplianceMetric(required_sections=["summary", "details"])
        sample = Sample(
            query="Explain Python.",
            context=["Python is a programming language."],
            answer="Python is a programming language.",
        )
        result = await metric.evaluate(sample)
        assert result.score < 1.0
        assert "summary" in result.details["missing_sections"]
        assert "details" in result.details["missing_sections"]

    @pytest.mark.asyncio
    async def test_combined_checks(self) -> None:
        metric = FormatComplianceMetric(
            max_length=50,
            forbidden_patterns=["sorry"],
            required_sections=["answer"],
        )
        sample = Sample(
            query="What is Python?",
            context=["Python is a programming language."],
            answer="Answer: Sorry, I don't know much about this very long topic that goes on and on forever.",
        )
        result = await metric.evaluate(sample)
        # Has multiple penalties
        assert result.score < 1.0
        assert result.details["length_penalty"] == 1
        assert len(result.details["matched_forbidden"]) > 0

    @pytest.mark.asyncio
    async def test_perfect_compliance_all_checks(self) -> None:
        metric = FormatComplianceMetric(
            max_length=200,
            forbidden_patterns=["sorry", "I don't know"],
            required_sections=["answer"],
        )
        sample = Sample(
            query="What is Python?",
            context=["Python is a programming language."],
            answer="Answer: Python is a popular programming language.",
        )
        result = await metric.evaluate(sample)
        assert result.score == 1.0
        assert result.details["length_penalty"] == 0
        assert result.details["matched_forbidden"] == []
        assert result.details["missing_sections"] == []

    @pytest.mark.asyncio
    async def test_default_no_checks_passes(self) -> None:
        metric = FormatComplianceMetric()
        sample = Sample(
            query="q",
            context=["c"],
            answer="anything goes here",
        )
        result = await metric.evaluate(sample)
        assert result.score == 1.0
