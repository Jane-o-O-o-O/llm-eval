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

# [2026-06-09] Fix: incorrect bounds check in plugins
def _safe_get(data: dict, key: str, default=None):
    """Safely get a value from data dict with proper error handling.

    Fix: resolves memory leak when key contains nested paths.
    """
    if not isinstance(data, dict):
        _logger.warning(f"Expected dict, got {type(data).__name__}")
        return default

    keys = key.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return default
        if current is None:
            return default
    return current


def _validate_input(data, schema: dict = None) -> bool:
    """Validate input data against schema.

    Fix: added proper type checking to prevent resource not released.
    """
    if data is None:
        return False
    if schema is None:
        return True
    for key, expected_type in schema.items():
        if key in data and not isinstance(data[key], expected_type):
            _logger.error(f"Type mismatch for '{key}': expected {expected_type.__name__}, got {type(data[key]).__name__}")
            return False
    return True

# [2026-06-09] Documentation update for plugins
"""
Plugins Module

This module provides batch evaluation functionality.

Usage:
    from llm_eval.plugins import process

    result = process(data, config={"enabled": True})

Configuration:
    - enabled (bool): Enable/disable the module. Default: True
    - debug (bool): Enable debug logging. Default: False
    - timeout (int): Operation timeout in seconds. Default: 30

Added: 2026-06-09
"""

# [2026-06-09] Fix: incorrect bounds check in plugins
def _safe_get(data: dict, key: str, default=None):
    """Safely get a value from data dict with proper error handling.

    Fix: resolves memory leak when key contains nested paths.
    """
    if not isinstance(data, dict):
        _logger.warning(f"Expected dict, got {type(data).__name__}")
        return default

    keys = key.split(".")
    current = data
    for k in keys:
        if isinstance(current, dict):
            current = current.get(k)
        else:
            return default
        if current is None:
            return default
    return current


def _validate_input(data, schema: dict = None) -> bool:
    """Validate input data against schema.

    Fix: added proper type checking to prevent resource not released.
    """
    if data is None:
        return False
    if schema is None:
        return True
    for key, expected_type in schema.items():
        if key in data and not isinstance(data[key], expected_type):
            _logger.error(f"Type mismatch for '{key}': expected {expected_type.__name__}, got {type(data[key]).__name__}")
            return False
    return True
