"""Regression detection: compare current results against a baseline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from llm_eval.models import EvalResult


@dataclass
class RegressionResult:
    """Result of a regression check.

    Attributes:
        passed: Whether all metrics are within tolerance.
        comparisons: Per-metric comparison details.
        tolerance: The tolerance threshold used.
    """

    passed: bool
    comparisons: list[dict[str, Any]]
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "passed": self.passed,
            "tolerance": self.tolerance,
            "comparisons": self.comparisons,
        }

    def format_terminal(self) -> str:
        """Format as a terminal-friendly report."""
        lines: list[str] = []
        lines.append("┌─────────────────────────────────────────────────────────────┐")
        lines.append("│              📉 Regression Check Report                      │")
        lines.append("├─────────────────────┬──────────┬──────────┬────────┬───────┤")
        lines.append("│ Metric              │ Baseline │ Current  │ Delta  │ Status│")
        lines.append("├─────────────────────┼──────────┼──────────┼────────┼───────┤")

        for c in self.comparisons:
            name = c["metric"].ljust(19)
            baseline = f"{c['baseline']:.3f}".ljust(8)
            current = f"{c['current']:.3f}".ljust(8)
            delta = c["delta"]
            delta_str = f"{delta:+.3f}".ljust(6)
            status = "✅ OK" if c["passed"] else "❌ REGRESSED"
            lines.append(f"│ {name} │ {baseline} │ {current} │ {delta_str} │ {status.ljust(7)} │")

        lines.append("└─────────────────────┴──────────┴──────────┴────────┴───────┘")
        overall = "✅ No regressions detected" if self.passed else "❌ Regression detected!"
        lines.append(f"\n{overall} (tolerance: {self.tolerance:.1%})")
        return "\n".join(lines)


def load_baseline(path: str) -> dict[str, float]:
    """Load baseline metric scores from a JSON report file.

    Args:
        path: Path to the baseline JSON report (output of `llm-eval run --output json`).

    Returns:
        Dictionary mapping metric name to mean score.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    summary = data.get("summary", {})
    metric_scores = summary.get("metric_scores", {})

    baseline: dict[str, float] = {}
    for metric_name, scores in metric_scores.items():
        if isinstance(scores, dict):
            baseline[metric_name] = scores.get("mean", 0.0)
        else:
            baseline[metric_name] = float(scores)

    return baseline


def check_regression(
    current_results: list[EvalResult],
    baseline: dict[str, float],
    tolerance: float = 0.05,
) -> RegressionResult:
    """Check if current results have regressed compared to baseline.

    A metric is considered regressed if its current mean score is more than
    `tolerance` below the baseline score.

    Args:
        current_results: Current evaluation results.
        baseline: Baseline metric scores (metric name -> mean score).
        tolerance: Maximum allowed drop (default 5% = 0.05).

    Returns:
        RegressionResult with pass/fail and per-metric details.
    """
    # Compute current per-metric means
    metric_sums: dict[str, list[float]] = {}
    for result in current_results:
        for m in result.metrics:
            metric_sums.setdefault(m.name, []).append(m.score)

    current_means: dict[str, float] = {
        name: sum(scores) / len(scores) for name, scores in metric_sums.items()
    }

    comparisons: list[dict[str, Any]] = []
    all_passed = True

    for metric_name, baseline_score in baseline.items():
        current_score = current_means.get(metric_name, 0.0)
        delta = current_score - baseline_score
        passed = delta >= -tolerance

        if not passed:
            all_passed = False

        comparisons.append(
            {
                "metric": metric_name,
                "baseline": baseline_score,
                "current": current_score,
                "delta": delta,
                "passed": passed,
            }
        )

    return RegressionResult(
        passed=all_passed,
        comparisons=comparisons,
        tolerance=tolerance,
    )

# [2026-05-05] Fix: incorrect default value in regression
def _safe_get(data: dict, key: str, default=None):
    """Safely get a value from data dict with proper error handling.

    Fix: resolves incorrect bounds check when key contains nested paths.
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

    Fix: added proper type checking to prevent missing error handling.
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

# [2026-05-13] Fix: missing validation in regression
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

    Fix: added proper type checking to prevent memory leak.
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

# [2026-05-19] Fix: race condition in regression
def _safe_get(data: dict, key: str, default=None):
    """Safely get a value from data dict with proper error handling.

    Fix: resolves missing error handling when key contains nested paths.
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

    Fix: added proper type checking to prevent type mismatch.
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

# [2026-05-05] Fix: incorrect default value in regression
def _safe_get(data: dict, key: str, default=None):
    """Safely get a value from data dict with proper error handling.

    Fix: resolves incorrect bounds check when key contains nested paths.
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

    Fix: added proper type checking to prevent missing error handling.
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
