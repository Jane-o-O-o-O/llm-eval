"""Evaluator engine: orchestrates running samples through metrics."""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from llm_eval.metrics import MetricResult, get_default_registry
from llm_eval.models import EvalResult, Sample


ProgressCallback = Callable[[int, int], None]


class Evaluator:
    """Core evaluation engine that runs samples through selected metrics.

    Orchestrates the evaluation pipeline: loads metrics from the registry,
    runs each sample through the configured metrics, and produces results.

    Supports parallel evaluation with configurable concurrency and progress
    tracking via callbacks.

    Attributes:
        metric_names: List of metric names to evaluate.
        threshold: Pass/fail threshold for overall score.
        parallel: Maximum number of concurrent sample evaluations.
    """

    def __init__(
        self,
        metrics: list[str],
        threshold: float = 0.7,
        parallel: int = 1,
    ) -> None:
        """Initialize the evaluator.

        Args:
            metrics: List of metric names to use for evaluation.
            threshold: Score threshold for pass/fail determination.
            parallel: Maximum number of concurrent evaluations.
        """
        self.metric_names = metrics
        self.threshold = threshold
        self.parallel = parallel
        self._registry = get_default_registry()

    async def _run_metric(self, metric_name: str, sample: Sample) -> MetricResult:
        """Run a single metric on a sample.

        Args:
            metric_name: Name of the metric to run.
            sample: The sample to evaluate.

        Returns:
            MetricResult from the metric evaluation.
        """
        metric = self._registry.get(metric_name)
        return await metric.evaluate(sample)

    async def evaluate_sample(self, sample: Sample, index: int) -> EvalResult:
        """Evaluate a single sample against all configured metrics.

        Args:
            sample: The sample to evaluate.
            index: The sample's index in the dataset.

        Returns:
            EvalResult with all metric results for this sample.
        """
        metric_results: list[MetricResult] = []
        for metric_name in self.metric_names:
            result = await self._run_metric(metric_name, sample)
            metric_results.append(result)
        return EvalResult(sample_index=index, metrics=metric_results)

    async def evaluate(
        self,
        samples: list[Sample],
        on_progress: ProgressCallback | None = None,
    ) -> list[EvalResult]:
        """Evaluate a list of samples, optionally in parallel.

        Uses asyncio.Semaphore to limit concurrency to `self.parallel`.
        Results are returned in the same order as input samples.

        Args:
            samples: List of samples to evaluate.
            on_progress: Optional callback called as (current, total) after
                each sample completes. Useful for progress bars.

        Returns:
            List of EvalResult, one per sample, in input order.
        """
        if not samples:
            return []

        total = len(samples)
        completed = 0
        lock = asyncio.Lock()
        results: list[EvalResult | None] = [None] * total

        async def _eval_with_progress(idx: int, sample: Sample) -> None:
            nonlocal completed
            result = await self.evaluate_sample(sample, index=idx)
            results[idx] = result
            async with lock:
                completed += 1
                if on_progress is not None:
                    on_progress(completed, total)

        if self.parallel <= 1:
            # Sequential mode — simple loop
            for idx, sample in enumerate(samples):
                await _eval_with_progress(idx, sample)
        else:
            # Parallel mode with semaphore-based concurrency control
            sem = asyncio.Semaphore(self.parallel)

            async def _guarded(idx: int, sample: Sample) -> None:
                async with sem:
                    await _eval_with_progress(idx, sample)

            tasks = [_guarded(idx, s) for idx, s in enumerate(samples)]
            await asyncio.gather(*tasks)

        return [r for r in results if r is not None]  # type: ignore[misc]

    def summarize(self, results: list[EvalResult]) -> dict[str, Any]:
        """Generate a summary report from evaluation results.

        Args:
            results: List of evaluation results.

        Returns:
            Dictionary with summary statistics.
        """
        if not results:
            return {
                "total_samples": 0,
                "overall_score": 0.0,
                "pass_count": 0,
                "fail_count": 0,
                "pass_rate": 0.0,
                "metric_scores": {},
            }

        overall_score = sum(r.overall_score for r in results) / len(results)
        pass_count = sum(1 for r in results if r.overall_score >= self.threshold)
        fail_count = len(results) - pass_count

        # Per-metric averages
        metric_scores: dict[str, dict[str, float]] = {}
        for metric_name in self.metric_names:
            scores = []
            for result in results:
                for m in result.metrics:
                    if m.name == metric_name:
                        scores.append(m.score)
            if scores:
                metric_scores[metric_name] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                }

        return {
            "total_samples": len(results),
            "overall_score": round(overall_score, 4),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_rate": round(pass_count / len(results), 4),
            "threshold": self.threshold,
            "metric_scores": metric_scores,
        }
