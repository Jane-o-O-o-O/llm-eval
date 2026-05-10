"""Core data models for llm-eval."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    """A single evaluation sample containing query, context, answer, and optional reference.

    Attributes:
        query: The user question or input.
        context: Retrieved context chunks (for RAG evaluations).
        answer: The generated answer to evaluate.
        reference: Ground truth answer (optional).
        metadata: Additional metadata for the sample.
    """

    query: str
    context: list[str]
    answer: str
    reference: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sample:
        """Create a Sample from a dictionary (e.g., parsed JSONL).

        Args:
            data: Dictionary with keys matching Sample fields.

        Returns:
            A new Sample instance.
        """
        return cls(
            query=data["query"],
            context=data["context"],
            answer=data["answer"],
            reference=data.get("reference"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class MetricResult:
    """Result of a single metric evaluation.

    Attributes:
        name: Name of the metric.
        score: Numeric score between 0.0 and 1.0.
        details: Additional details about the evaluation.
    """

    name: str
    score: float
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a serializable dictionary.

        Returns:
            Dictionary representation of the metric result.
        """
        return {
            "name": self.name,
            "score": self.score,
            "details": self.details,
        }


@dataclass
class EvalResult:
    """Result of evaluating a single sample across all metrics.

    Attributes:
        sample_index: Index of the evaluated sample.
        metrics: List of metric results.
    """

    sample_index: int
    metrics: list[MetricResult] = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        """Calculate the average score across all metrics.

        Returns:
            Mean of all metric scores, or 0.0 if no metrics.
        """
        if not self.metrics:
            return 0.0
        return sum(m.score for m in self.metrics) / len(self.metrics)

    def to_dict(self) -> dict[str, Any]:
        """Convert to a serializable dictionary.

        Returns:
            Dictionary representation of the evaluation result.
        """
        return {
            "sample_index": self.sample_index,
            "overall_score": self.overall_score,
            "metrics": [m.to_dict() for m in self.metrics],
        }


@dataclass
class JudgeConfig:
    """Configuration for the LLM judge.

    Attributes:
        model: Model identifier (e.g., 'gpt-4o', 'claude-3-opus').
        base_url: Custom API endpoint URL.
        temperature: Sampling temperature.
        max_retries: Maximum retry attempts on failure.
        timeout: Request timeout in seconds.
    """

    model: str = "gpt-4o"
    base_url: str | None = None
    temperature: float = 0.0
    max_retries: int = 3
    timeout: int = 60


@dataclass
class EvalConfig:
    """Top-level evaluation configuration.

    Attributes:
        judge: Judge model configuration.
        evaluations: List of evaluation definitions.
        threshold: Default pass/fail threshold.
        output_format: Output format (terminal, json, csv, html).
    """

    judge: JudgeConfig = field(default_factory=JudgeConfig)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    threshold: float = 0.7
    output_format: str = "terminal"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalConfig:
        """Create an EvalConfig from a parsed YAML dictionary.

        Args:
            data: Parsed configuration dictionary.

        Returns:
            A new EvalConfig instance.
        """
        judge_data = data.get("judge", {})
        judge = JudgeConfig(
            model=judge_data.get("model", "gpt-4o"),
            base_url=judge_data.get("base_url"),
            temperature=judge_data.get("temperature", 0.0),
            max_retries=judge_data.get("max_retries", 3),
            timeout=judge_data.get("timeout", 60),
        )
        defaults = data.get("defaults", {})
        return cls(
            judge=judge,
            evaluations=data.get("evaluations", []),
            threshold=defaults.get("threshold", 0.7),
            output_format=defaults.get("output_format", "terminal"),
        )
