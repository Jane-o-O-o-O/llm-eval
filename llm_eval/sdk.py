"""Python SDK for programmatic evaluation without the CLI.

Usage::

    from llm_eval.sdk import evaluate, evaluate_file

    # From in-memory samples
    results = await evaluate(
        samples=[{"query": "What?", "context": ["..."], "answer": "..."}],
        metrics=["faithfulness", "answer_relevancy"],
        model="gpt-4o",
    )

    # From a dataset file
    results = await evaluate_file(
        path="samples.jsonl",
        metrics=["faithfulness"],
        config="evals.yaml",
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llm_eval.dataset import load_dataset
from llm_eval.evaluator import Evaluator
from llm_eval.models import EvalConfig, JudgeConfig, Sample
from llm_eval.report import format_json_report, format_terminal_report


@dataclass
class EvalOutput:
    """Structured output from the SDK evaluate functions.

    Attributes:
        results: Raw evaluation results.
        summary: Summary statistics dict.
        terminal: Pre-formatted terminal report string.
        json: Pre-formatted JSON report string.
    """

    results: list[Any] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    terminal: str = ""
    json: str = ""

    @property
    def overall_score(self) -> float:
        """Get the overall weighted score."""
        return self.summary.get("overall_score", 0.0)

    @property
    def passed(self) -> bool:
        """Check if all samples passed the threshold."""
        return self.summary.get("fail_count", 0) == 0 and self.summary.get("total_samples", 0) > 0

    @property
    def total_samples(self) -> int:
        """Number of evaluated samples."""
        return self.summary.get("total_samples", 0)


async def evaluate(
    samples: list[dict[str, Any]],
    metrics: list[str],
    *,
    model: str = "gpt-4o",
    base_url: str | None = None,
    threshold: float = 0.7,
    parallel: int = 1,
    metric_weights: dict[str, float] | None = None,
    temperature: float = 0.0,
    timeout: int = 60,
) -> EvalOutput:
    """Evaluate samples programmatically.

    Args:
        samples: List of sample dicts with keys: query, context, answer.
                 Optional: reference, metadata.
        metrics: List of metric names to evaluate (e.g. ["faithfulness"]).
        model: Judge model identifier.
        base_url: Custom API endpoint (for non-OpenAI providers).
        threshold: Pass/fail threshold.
        parallel: Number of concurrent evaluations.
        metric_weights: Optional per-metric weights.
        temperature: Judge temperature.
        timeout: HTTP request timeout in seconds.

    Returns:
        EvalOutput with results, summary, and formatted reports.
    """
    parsed = [Sample.from_dict(s) for s in samples]
    judge_config = JudgeConfig(
        model=model,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout,
    )
    evaluator = Evaluator(
        metrics=metrics,
        threshold=threshold,
        parallel=parallel,
        metric_weights=metric_weights,
        judge_config=judge_config,
    )
    results = await evaluator.evaluate(parsed)
    summary = evaluator.summarize(results)
    return EvalOutput(
        results=results,
        summary=summary,
        terminal=format_terminal_report(results, summary),
        json=format_json_report(results, summary),
    )


async def evaluate_file(
    path: str,
    metrics: list[str] | None = None,
    *,
    config: str | None = None,
    model: str = "gpt-4o",
    base_url: str | None = None,
    threshold: float = 0.7,
    parallel: int = 1,
) -> EvalOutput:
    """Evaluate from a dataset file, optionally with a YAML config.

    When *config* is provided, metrics and judge settings are read from the
    config file (CLI-style). Explicit keyword arguments override config values.

    Args:
        path: Path to a JSONL or CSV dataset file.
        metrics: Metric names (ignored if config provides them).
        config: Optional YAML config file path.
        model: Judge model (overridden by config).
        base_url: Custom API endpoint.
        threshold: Pass/fail threshold.
        parallel: Number of concurrent evaluations.

    Returns:
        EvalOutput with results, summary, and formatted reports.
    """
    samples = load_dataset(path)

    if config:
        import yaml

        with open(config, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg = EvalConfig.from_dict(raw)
        judge = cfg.judge
        eval_def = cfg.evaluations[0] if cfg.evaluations else {}
        metrics = metrics or eval_def.get("metrics", [])
        threshold = threshold if threshold != 0.7 else eval_def.get("threshold", cfg.threshold)
        parallel = parallel if parallel != 1 else eval_def.get("parallel", cfg.parallel)
    else:
        judge = JudgeConfig(model=model, base_url=base_url)

    evaluator = Evaluator(
        metrics=metrics or [],
        threshold=threshold,
        parallel=parallel,
        judge_config=judge,
    )
    results = await evaluator.evaluate(samples)
    summary = evaluator.summarize(results)
    return EvalOutput(
        results=results,
        summary=summary,
        terminal=format_terminal_report(results, summary),
        json=format_json_report(results, summary),
    )


__all__ = ["evaluate", "evaluate_file", "EvalOutput"]
