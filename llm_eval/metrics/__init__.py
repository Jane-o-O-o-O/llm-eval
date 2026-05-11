"""Metric framework: base class and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llm_eval.models import MetricResult, Sample


class Metric(ABC):
    """Abstract base class for all evaluation metrics.

    Subclasses must define `name`, `description`, and implement `evaluate()`.
    """

    name: str
    description: str

    @abstractmethod
    def evaluate(self, sample: Sample) -> MetricResult:
        """Evaluate a single sample and return a metric result.

        Args:
            sample: The evaluation sample to score.

        Returns:
            A MetricResult with the score and optional details.
        """


class MetricRegistry:
    """Registry for discovering and retrieving metric implementations.

    Metrics are registered by name and retrieved for use during evaluation runs.
    """

    def __init__(self) -> None:
        self._metrics: dict[str, Metric] = {}

    def register(self, metric: Metric) -> None:
        """Register a metric instance by its name.

        Args:
            metric: A Metric subclass instance to register.
        """
        self._metrics[metric.name] = metric

    def get(self, name: str) -> Metric:
        """Retrieve a registered metric by name.

        Args:
            name: The metric name.

        Returns:
            The registered Metric instance.

        Raises:
            KeyError: If the metric is not registered.
        """
        if name not in self._metrics:
            raise KeyError(f"Metric not registered: {name}")
        return self._metrics[name]

    def __contains__(self, name: str) -> bool:
        """Support `in` operator for checking metric registration."""
        return name in self._metrics

    def is_registered(self, name: str) -> bool:
        """Check if a metric is registered.

        Args:
            name: The metric name.

        Returns:
            True if the metric is registered.
        """
        return name in self._metrics

    def list_metrics(self) -> list[str]:
        """List all registered metric names.

        Returns:
            Sorted list of metric names.
        """
        return sorted(self._metrics.keys())


def get_default_registry() -> MetricRegistry:
    """Create a registry pre-loaded with all built-in metrics.

    Returns:
        A MetricRegistry with all built-in metrics registered.
    """
    from llm_eval.metrics.faithfulness import FaithfulnessMetric
    from llm_eval.metrics.answer_relevancy import AnswerRelevancyMetric
    from llm_eval.metrics.context_precision import ContextPrecisionMetric
    from llm_eval.metrics.context_recall import ContextRecallMetric
    from llm_eval.metrics.format_compliance import FormatComplianceMetric

    registry = MetricRegistry()
    registry.register(FaithfulnessMetric())
    registry.register(AnswerRelevancyMetric())
    registry.register(ContextPrecisionMetric())
    registry.register(ContextRecallMetric())
    registry.register(FormatComplianceMetric())
    return registry


# Re-export MetricResult for convenience
__all__ = ["Metric", "MetricResult", "MetricRegistry", "get_default_registry"]
