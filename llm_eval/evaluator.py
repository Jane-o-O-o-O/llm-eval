"""Evaluator engine: orchestrates running samples through metrics."""

from __future__ import annotations

import asyncio
from typing import Any

from llm_eval.metrics import MetricResult, get_default_registry
from llm_eval.models import EvalResult, JudgeConfig, Sample


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Calculate percentile from pre-sorted data using linear interpolation.

    Args:
        sorted_data: Pre-sorted list of values.
        pct: Percentile to calculate (0-100).

    Returns:
        The percentile value.
    """
    if not sorted_data:
        return 0.0
    if len(sorted_data) == 1:
        return sorted_data[0]
    k = (len(sorted_data) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    d = k - f
    return sorted_data[f] + d * (sorted_data[c] - sorted_data[f])


class Evaluator:
    """Core evaluation engine that runs samples through selected metrics.

    Orchestrates the evaluation pipeline: loads metrics from the registry,
    runs each sample through the configured metrics, and produces results.
    Supports parallel evaluation with configurable concurrency.

    Attributes:
        metric_names: List of metric names to evaluate.
        threshold: Pass/fail threshold for overall score.
        parallel: Number of concurrent evaluations.
    """

    def __init__(
        self,
        metrics: list[str],
        threshold: float = 0.7,
        parallel: int = 1,
        metric_weights: dict[str, float] | None = None,
        judge_config: JudgeConfig | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            metrics: List of metric names to use for evaluation.
            threshold: Score threshold for pass/fail determination.
            parallel: Number of concurrent evaluations (1 = sequential).
            metric_weights: Optional per-metric weights for overall score.
                            If empty or None, uses equal weights.
            judge_config: Optional judge model configuration to pass to metrics.
        """
        self.metric_names = metrics
        self.threshold = threshold
        self.parallel = max(1, parallel)
        self.metric_weights = metric_weights or {}
        self._registry = get_default_registry(judge_config=judge_config)

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
        progress_callback: Any | None = None,
    ) -> list[EvalResult]:
        """Evaluate a list of samples, with optional parallel execution.

        Args:
            samples: List of samples to evaluate.
            progress_callback: Optional callback called after each sample completes.
                               Signature: callback(completed: int, total: int)

        Returns:
            List of EvalResult, one per sample.
        """
        if self.parallel <= 1:
            # Sequential evaluation
            results: list[EvalResult] = []
            for i, sample in enumerate(samples):
                result = await self.evaluate_sample(sample, index=i)
                results.append(result)
                if progress_callback:
                    progress_callback(len(results), len(samples))
            return results

        # Parallel evaluation with semaphore
        semaphore = asyncio.Semaphore(self.parallel)
        results_list: list[EvalResult | None] = [None] * len(samples)
        completed_count = 0

        async def _eval_with_semaphore(idx: int, sample: Sample) -> None:
            nonlocal completed_count
            async with semaphore:
                result = await self.evaluate_sample(sample, index=idx)
                results_list[idx] = result
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, len(samples))

        tasks = [_eval_with_semaphore(i, sample) for i, sample in enumerate(samples)]
        await asyncio.gather(*tasks)
        return [r for r in results_list if r is not None]

    def summarize(self, results: list[EvalResult]) -> dict[str, Any]:
        """Generate a summary report from evaluation results.

        Uses metric_weights (if configured) for computing the weighted overall score.
        Falls back to equal weights when no weights are specified.
        Includes distribution statistics: median, p25, p75, std_dev, min, max.

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
                "median": 0.0,
                "p25": 0.0,
                "p75": 0.0,
                "std_dev": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
            }

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

        # Weighted overall score
        if self.metric_weights:
            total_weight = 0.0
            weighted_sum = 0.0
            for metric_name, scores in metric_scores.items():
                w = self.metric_weights.get(metric_name, 1.0)
                weighted_sum += scores["mean"] * w
                total_weight += w
            overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            overall_score = sum(r.overall_score for r in results) / len(results)

        pass_count = sum(1 for r in results if r.overall_score >= self.threshold)
        fail_count = len(results) - pass_count

        # Distribution statistics
        all_scores = sorted(r.overall_score for r in results)
        n = len(all_scores)
        median = _percentile(all_scores, 50)
        p25 = _percentile(all_scores, 25)
        p75 = _percentile(all_scores, 75)
        mean = sum(all_scores) / n
        variance = sum((s - mean) ** 2 for s in all_scores) / n
        std_dev = variance ** 0.5

        return {
            "total_samples": len(results),
            "overall_score": round(overall_score, 4),
            "pass_count": pass_count,
            "fail_count": fail_count,
            "pass_rate": round(pass_count / len(results), 4),
            "threshold": self.threshold,
            "metric_scores": metric_scores,
            "median": round(median, 4),
            "p25": round(p25, 4),
            "p75": round(p75, 4),
            "std_dev": round(std_dev, 4),
            "min_score": round(all_scores[0], 4),
            "max_score": round(all_scores[-1], 4),
        }
