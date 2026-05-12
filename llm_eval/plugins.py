"""Custom metric plugin loader for external metric implementations."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from llm_eval.metrics import Metric, MetricRegistry

logger = logging.getLogger(__name__)


def load_custom_metrics(
    registry: MetricRegistry,
    custom_metrics: list[dict[str, Any]],
) -> list[str]:
    """Load and register custom metrics from module paths.

    Each entry in custom_metrics should be a dict with:
        - module: Python module path (e.g., "my_package.metrics")
        - class: Class name within the module (e.g., "MyCustomMetric")

    Args:
        registry: The MetricRegistry to register metrics into.
        custom_metrics: List of custom metric specifications.

    Returns:
        List of successfully loaded metric names.

    Raises:
        ImportError: If a module cannot be imported.
        AttributeError: If the class is not found in the module.
    """
    loaded: list[str] = []

    for spec in custom_metrics:
        module_path = spec.get("module", "")
        class_name = spec.get("class", "")

        if not module_path or not class_name:
            logger.warning("Skipping custom metric with missing module/class: %s", spec)
            continue

        try:
            module = importlib.import_module(module_path)
            metric_cls = getattr(module, class_name)
            instance = metric_cls()

            if not isinstance(instance, Metric):
                raise TypeError(
                    f"{class_name} must be a subclass of Metric, got {type(instance).__name__}"
                )

            registry.register(instance)
            loaded.append(instance.name)
            logger.info("Loaded custom metric: %s from %s", instance.name, module_path)

        except (ImportError, AttributeError, TypeError) as exc:
            logger.error("Failed to load custom metric %s.%s: %s", module_path, class_name, exc)
            raise

    return loaded
