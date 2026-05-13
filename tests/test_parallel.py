"""Tests for parallel evaluation and progress callbacks."""

from unittest.mock import AsyncMock, patch

import pytest

from llm_eval.evaluator import Evaluator
from llm_eval.metrics import MetricResult
from llm_eval.models import Sample


def _make_sample(i: int) -> Sample:
    return Sample(query=f"q{i}", context=[f"c{i}"], answer=f"a{i}")


class TestParallelEvaluation:
    """Tests for parallel/batch evaluation with progress tracking."""

    @pytest.mark.asyncio
    async def test_parallel_produces_correct_results(self) -> None:
        """Parallel evaluation should produce the same results as sequential."""
        evaluator = Evaluator(metrics=["faithfulness"], parallel=1)
        samples = [_make_sample(i) for i in range(5)]

        mock_result = MetricResult(name="faithfulness", score=0.8, details={})
        with patch.object(
            evaluator, "_run_metric", new_callable=AsyncMock, return_value=mock_result
        ):
            results = await evaluator.evaluate(samples)
            assert len(results) == 5
            for i, r in enumerate(results):
                assert r.sample_index == i
                assert r.metrics[0].score == 0.8

    @pytest.mark.asyncio
    async def test_parallel_with_concurrency(self) -> None:
        """Parallel evaluation with concurrency > 1 should still produce correct results."""
        evaluator = Evaluator(metrics=["faithfulness"], parallel=3)
        samples = [_make_sample(i) for i in range(6)]

        call_count = 0

        async def mock_metric(name, sample):
            nonlocal call_count
            call_count += 1
            return MetricResult(name="faithfulness", score=0.7, details={"call": call_count})

        with patch.object(evaluator, "_run_metric", side_effect=mock_metric):
            results = await evaluator.evaluate(samples)
            assert len(results) == 6
            # All samples should have been evaluated
            assert call_count == 6

    @pytest.mark.asyncio
    async def test_progress_callback_is_called(self) -> None:
        """A progress callback should be called once per sample."""
        evaluator = Evaluator(metrics=["faithfulness"], parallel=1)
        samples = [_make_sample(i) for i in range(3)]

        progress_calls = []

        def on_progress(current: int, total: int):
            progress_calls.append((current, total))

        mock_result = MetricResult(name="faithfulness", score=0.9, details={})
        with patch.object(
            evaluator, "_run_metric", new_callable=AsyncMock, return_value=mock_result
        ):
            await evaluator.evaluate(samples, progress_callback=on_progress)

        assert len(progress_calls) == 3
        # Should be (1,3), (2,3), (3,3)
        assert progress_calls[-1] == (3, 3)

    @pytest.mark.asyncio
    async def test_progress_callback_not_required(self) -> None:
        """Evaluation should work without a progress callback."""
        evaluator = Evaluator(metrics=["faithfulness"], parallel=1)
        samples = [_make_sample(0)]

        mock_result = MetricResult(name="faithfulness", score=0.9, details={})
        with patch.object(
            evaluator, "_run_metric", new_callable=AsyncMock, return_value=mock_result
        ):
            results = await evaluator.evaluate(samples)
            assert len(results) == 1

    @pytest.mark.asyncio
    async def test_empty_samples_with_progress(self) -> None:
        """Evaluating empty samples list with callback should not call callback."""
        evaluator = Evaluator(metrics=["faithfulness"], parallel=1)
        progress_calls = []

        def on_progress(current: int, total: int):
            progress_calls.append((current, total))

        results = await evaluator.evaluate([], progress_callback=on_progress)
        assert len(results) == 0
        assert len(progress_calls) == 0

    def test_parallel_parameter_stored(self) -> None:
        """The parallel parameter should be stored on the evaluator."""
        evaluator = Evaluator(metrics=["faithfulness"], parallel=5)
        assert evaluator.parallel == 5
