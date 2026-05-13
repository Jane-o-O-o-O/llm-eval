"""Tests for the Python SDK (llm_eval.sdk)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from llm_eval.models import EvalResult, MetricResult
from llm_eval.sdk import EvalOutput, evaluate, evaluate_file


@pytest.fixture
def mock_results():
    """Return mock evaluation results."""
    return [
        EvalResult(
            sample_index=0,
            metrics=[MetricResult(name="faithfulness", score=0.9, details={})],
        ),
        EvalResult(
            sample_index=1,
            metrics=[MetricResult(name="faithfulness", score=0.8, details={})],
        ),
    ]


@pytest.fixture
def sample_dicts():
    """Return sample data as dicts."""
    return [
        {"query": "What is Python?", "context": ["Python is a language."], "answer": "A language."},
        {"query": "What is Rust?", "context": ["Rust is a systems language."], "answer": "A systems language."},
    ]


class TestEvalOutput:
    """Tests for EvalOutput dataclass."""

    def test_overall_score_from_summary(self):
        output = EvalOutput(summary={"overall_score": 0.85})
        assert output.overall_score == 0.85

    def test_overall_score_empty(self):
        output = EvalOutput()
        assert output.overall_score == 0.0

    def test_passed_when_no_failures(self):
        output = EvalOutput(summary={"fail_count": 0, "total_samples": 5})
        assert output.passed is True

    def test_not_passed_when_failures(self):
        output = EvalOutput(summary={"fail_count": 2, "total_samples": 5})
        assert output.passed is False

    def test_not_passed_when_empty(self):
        output = EvalOutput(summary={"fail_count": 0, "total_samples": 0})
        assert output.passed is False

    def test_total_samples(self):
        output = EvalOutput(summary={"total_samples": 10})
        assert output.total_samples == 10

    def test_total_samples_default(self):
        output = EvalOutput()
        assert output.total_samples == 0


class TestEvaluate:
    """Tests for the evaluate() async function."""

    @pytest.mark.asyncio
    async def test_basic_evaluate(self, sample_dicts, mock_results):
        with patch("llm_eval.sdk.Evaluator") as mock_evaluator:
            instance = mock_evaluator.return_value
            instance.evaluate = AsyncMock(return_value=mock_results)
            instance.summarize.return_value = {
                "total_samples": 2,
                "overall_score": 0.85,
                "pass_count": 2,
                "fail_count": 0,
                "pass_rate": 1.0,
                "metric_scores": {"faithfulness": {"mean": 0.85, "min": 0.8, "max": 0.9}},
            }

            output = await evaluate(
                samples=sample_dicts,
                metrics=["faithfulness"],
                model="gpt-4o",
            )

            assert isinstance(output, EvalOutput)
            assert output.overall_score == 0.85
            assert output.passed is True
            assert output.total_samples == 2
            assert output.terminal  # has content
            assert output.json  # has content

    @pytest.mark.asyncio
    async def test_evaluate_with_custom_params(self, sample_dicts, mock_results):
        with patch("llm_eval.sdk.Evaluator") as mock_evaluator:
            instance = mock_evaluator.return_value
            instance.evaluate = AsyncMock(return_value=mock_results)
            instance.summarize.return_value = {"overall_score": 0.5, "total_samples": 2, "fail_count": 1, "pass_count": 1, "pass_rate": 0.5, "metric_scores": {}}

            output = await evaluate(
                samples=sample_dicts,
                metrics=["faithfulness"],
                model="claude-3",
                base_url="https://custom.api/v1",
                threshold=0.9,
                parallel=3,
            )
            assert output.summary["overall_score"] == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_empty_samples(self):
        with patch("llm_eval.sdk.Evaluator") as mock_evaluator:
            instance = mock_evaluator.return_value
            instance.evaluate = AsyncMock(return_value=[])
            instance.summarize.return_value = {"total_samples": 0, "overall_score": 0.0, "pass_count": 0, "fail_count": 0, "pass_rate": 0.0, "metric_scores": {}}

            output = await evaluate(samples=[], metrics=["faithfulness"])
            assert output.total_samples == 0
            assert output.overall_score == 0.0


class TestEvaluateFile:
    """Tests for the evaluate_file() async function."""

    @pytest.mark.asyncio
    async def test_evaluate_from_jsonl(self, tmp_path, mock_results):
        # Create a temp JSONL file
        ds = tmp_path / "samples.jsonl"
        ds.write_text(
            '{"query": "Q1", "context": ["C1"], "answer": "A1"}\n'
            '{"query": "Q2", "context": ["C2"], "answer": "A2"}\n'
        )

        with patch("llm_eval.sdk.Evaluator") as mock_evaluator:
            instance = mock_evaluator.return_value
            instance.evaluate = AsyncMock(return_value=mock_results)
            instance.summarize.return_value = {"overall_score": 0.8, "total_samples": 2, "fail_count": 0, "pass_count": 2, "pass_rate": 1.0, "metric_scores": {}}

            output = await evaluate_file(
                path=str(ds),
                metrics=["faithfulness"],
                model="gpt-4o",
            )
            assert output.total_samples == 2

    @pytest.mark.asyncio
    async def test_evaluate_from_csv(self, tmp_path, mock_results):
        ds = tmp_path / "samples.csv"
        ds.write_text("query,context,answer\nQ1,C1,A1\nQ2,C2,A2\n")

        with patch("llm_eval.sdk.Evaluator") as mock_evaluator:
            instance = mock_evaluator.return_value
            instance.evaluate = AsyncMock(return_value=mock_results)
            instance.summarize.return_value = {"overall_score": 0.8, "total_samples": 2, "fail_count": 0, "pass_count": 2, "pass_rate": 1.0, "metric_scores": {}}

            output = await evaluate_file(path=str(ds), metrics=["faithfulness"])
            assert output.total_samples == 2

    @pytest.mark.asyncio
    async def test_evaluate_file_with_config(self, tmp_path, mock_results):
        ds = tmp_path / "samples.jsonl"
        ds.write_text('{"query": "Q", "context": ["C"], "answer": "A"}\n')
        cfg = tmp_path / "evals.yaml"
        cfg.write_text(
            "judge:\n  model: gpt-4o\n  temperature: 0\n"
            "defaults:\n  threshold: 0.7\n  output_format: terminal\n"
            "evaluations:\n  - name: test\n    dataset: samples.jsonl\n    metrics:\n      - faithfulness\n"
        )

        with patch("llm_eval.sdk.Evaluator") as mock_evaluator:
            instance = mock_evaluator.return_value
            instance.evaluate = AsyncMock(return_value=mock_results)
            instance.summarize.return_value = {"overall_score": 0.9, "total_samples": 1, "fail_count": 0, "pass_count": 1, "pass_rate": 1.0, "metric_scores": {}}

            output = await evaluate_file(
                path=str(ds),
                config=str(cfg),
            )
            assert output.overall_score == 0.9

    @pytest.mark.asyncio
    async def test_evaluate_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            await evaluate_file(path="/nonexistent/file.jsonl", metrics=["faithfulness"])
