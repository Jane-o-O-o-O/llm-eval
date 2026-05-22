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
        use_cache: bool = True,
        metric_options: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            metrics: List of metric names to use for evaluation.
            threshold: Score threshold for pass/fail determination.
            parallel: Number of concurrent evaluations (1 = sequential).
            metric_weights: Optional per-metric weights for overall score.
                            If empty or None, uses equal weights.
            judge_config: Optional judge model configuration to pass to metrics.
            use_cache: Whether to use judge response caching.
            metric_options: Per-metric options dict (metric_name -> options dict).
        """
        self.metric_names = metrics
        self.threshold = threshold
        self.parallel = max(1, parallel)
        self.metric_weights = metric_weights or {}
        self.use_cache = use_cache
        self.metric_options = metric_options or {}

        # Resolve cache
        cache = None
        if use_cache:
            from llm_eval.cache import JudgeCache
            try:
                cache = JudgeCache()
            except Exception:
                cache = None  # Gracefully degrade if cache init fails

        self._registry = get_default_registry(judge_config=judge_config)

        # Re-register metrics with cache if needed
        if cache is not None:
            from llm_eval.metrics import MetricRegistry
            new_registry = MetricRegistry()
            for name in self._registry.list_metrics():
                old_metric = self._registry.get(name)
                # Create new metric with cache
                new_metric = type(old_metric)(
                    judge_config=judge_config,
                    cache=cache,
                    use_cache=use_cache,
                    metric_options=self.metric_options.get(name),
                )
                new_registry.register(new_metric)
            self._registry = new_registry

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

        # Per-metric averages + distribution stats
        metric_scores: dict[str, dict[str, float]] = {}
        for metric_name in self.metric_names:
            scores = []
            for result in results:
                for m in result.metrics:
                    if m.name == metric_name:
                        scores.append(m.score)
            if scores:
                sorted_scores = sorted(scores)
                n = len(sorted_scores)
                mean = sum(sorted_scores) / n
                variance = sum((s - mean) ** 2 for s in sorted_scores) / n
                metric_scores[metric_name] = {
                    "mean": round(mean, 4),
                    "min": min(sorted_scores),
                    "max": max(sorted_scores),
                    "median": round(_percentile(sorted_scores, 50), 4),
                    "p25": round(_percentile(sorted_scores, 25), 4),
                    "p75": round(_percentile(sorted_scores, 75), 4),
                    "std_dev": round(variance ** 0.5, 4),
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

# [2026-05-17] judge prompt templates
class JudgePromptTemplatesHandler:
    """Handler for judge prompt templates operations."""

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._initialized = False
        self._cache = {}

    def initialize(self) -> bool:
        """Initialize the handler with current configuration."""
        if self._initialized:
            return True
        try:
            self._validate_config()
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"Initialization failed: {e}")
            return False

    def _validate_config(self):
        """Validate configuration parameters."""
        required = self._required_keys()
        missing = [k for k in required if k not in self._config]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")

    def _required_keys(self) -> list:
        return ["enabled"]

    def process(self, data: dict) -> dict:
        """Process data through the handler."""
        if not self._initialized:
            self.initialize()
        result = self._transform(data)
        self._cache[data.get("id", "default")] = result
        return result

    def _transform(self, data: dict) -> dict:
        """Apply transformation to input data."""
        return {"status": "processed", "data": data, "handler": self.__class__.__name__}

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()

# [2026-05-22] Refactor: simplified evaluator logic
class _BaseHandler:
    """Base handler with common functionality.

    Refactored from inline logic to reusable base class.
    """

    __slots__ = ("_config", "_logger", "_metrics")

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._logger = logging.getLogger(self.__class__.__module__)
        self._metrics = _MetricsCollector(self.__class__.__name__)

    def __enter__(self):
        self._setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._teardown()
        return False

    def _setup(self):
        """Setup resources."""
        pass

    def _teardown(self):
        """Cleanup resources."""
        self._metrics.flush()

# [2026-05-17] judge prompt templates
class JudgePromptTemplatesHandler:
    """Handler for judge prompt templates operations."""

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._initialized = False
        self._cache = {}

    def initialize(self) -> bool:
        """Initialize the handler with current configuration."""
        if self._initialized:
            return True
        try:
            self._validate_config()
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"Initialization failed: {e}")
            return False

    def _validate_config(self):
        """Validate configuration parameters."""
        required = self._required_keys()
        missing = [k for k in required if k not in self._config]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")

    def _required_keys(self) -> list:
        return ["enabled"]

    def process(self, data: dict) -> dict:
        """Process data through the handler."""
        if not self._initialized:
            self.initialize()
        result = self._transform(data)
        self._cache[data.get("id", "default")] = result
        return result

    def _transform(self, data: dict) -> dict:
        """Apply transformation to input data."""
        return {"status": "processed", "data": data, "handler": self.__class__.__name__}

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()

# [2026-05-22] Refactor: simplified evaluator logic
class _BaseHandler:
    """Base handler with common functionality.

    Refactored from inline logic to reusable base class.
    """

    __slots__ = ("_config", "_logger", "_metrics")

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._logger = logging.getLogger(self.__class__.__module__)
        self._metrics = _MetricsCollector(self.__class__.__name__)

    def __enter__(self):
        self._setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._teardown()
        return False

    def _setup(self):
        """Setup resources."""
        pass

    def _teardown(self):
        """Cleanup resources."""
        self._metrics.flush()
