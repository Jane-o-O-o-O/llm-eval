"""Compare evaluation results across runs or models."""

from __future__ import annotations

import json
from typing import Any


def load_report(path: str) -> dict[str, Any]:
    """Load a JSON evaluation report.

    Args:
        path: Path to the JSON report file.

    Returns:
        Parsed report dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON or lacks expected fields.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "summary" not in data:
        raise ValueError(f"Report missing 'summary' field: {path}")

    return data


def compare_reports(
    report_a: dict[str, Any],
    report_b: dict[str, Any],
    label_a: str = "Report A",
    label_b: str = "Report B",
    path_a: str = "",
    path_b: str = "",
) -> dict[str, Any]:
    """Compare two evaluation reports and generate a diff.

    Args:
        report_a: First report (baseline).
        report_b: Second report (current).
        label_a: Label for the first report.
        label_b: Label for the second report.
        path_a: Optional file path of report A.
        path_b: Optional file path of report B.

    Returns:
        Comparison dictionary with per-metric diffs.
    """
    summary_a = report_a.get("summary", {})
    summary_b = report_b.get("summary", {})

    metrics_a = summary_a.get("metric_scores", {})
    metrics_b = summary_b.get("metric_scores", {})

    all_metrics = sorted(set(metrics_a.keys()) | set(metrics_b.keys()))

    comparisons: list[dict[str, Any]] = []
    for metric in all_metrics:
        score_a = metrics_a.get(metric, {}).get("mean", 0.0) if isinstance(metrics_a.get(metric), dict) else float(metrics_a.get(metric, 0.0))
        score_b = metrics_b.get(metric, {}).get("mean", 0.0) if isinstance(metrics_b.get(metric), dict) else float(metrics_b.get(metric, 0.0))
        delta = score_b - score_a

        comparisons.append({
            "metric": metric,
            label_a: round(score_a, 4),
            label_b: round(score_b, 4),
            "delta": round(delta, 4),
        })

    overall_a = summary_a.get("overall_score", 0.0)
    overall_b = summary_b.get("overall_score", 0.0)

    return {
        label_a: {"path": path_a, "overall": overall_a},
        label_b: {"path": path_b, "overall": overall_b},
        "overall_delta": round(overall_b - overall_a, 4),
        "comparisons": comparisons,
    }


def format_terminal_comparison(comparison: dict[str, Any]) -> str:
    """Format a comparison as a terminal-friendly table.

    Args:
        comparison: Comparison dictionary from compare_reports.

    Returns:
        Formatted string for terminal display.
    """
    labels = [k for k in comparison if k not in ("overall_delta", "comparisons")]
    label_a = labels[0] if len(labels) > 0 else "Report A"
    label_b = labels[1] if len(labels) > 1 else "Report B"

    lines: list[str] = []
    lines.append("┌──────────────────────────────────────────────────────────────────┐")
    lines.append("│                    📊 Evaluation Comparison                       │")
    lines.append("├─────────────────────┬──────────────┬──────────────┬──────────────┤")
    header = f"│ {'Metric'.ljust(19)} │ {label_a.ljust(12)} │ {label_b.ljust(12)} │ {'Delta'.ljust(12)} │"
    lines.append(header)
    lines.append("├─────────────────────┼──────────────┼──────────────┼──────────────┤")

    for c in comparison.get("comparisons", []):
        name = c["metric"].ljust(19)
        val_a = f"{c[label_a]:.4f}".ljust(12)
        val_b = f"{c[label_b]:.4f}".ljust(12)
        delta = c["delta"]
        delta_str = f"{delta:+.4f}".ljust(12)
        lines.append(f"│ {name} │ {val_a} │ {val_b} │ {delta_str} │")

    lines.append("├─────────────────────┼──────────────┼──────────────┼──────────────┤")
    overall_a = comparison[label_a]["overall"]
    overall_b = comparison[label_b]["overall"]
    overall_delta = comparison["overall_delta"]
    lines.append(
        f"│ {'Overall'.ljust(19)} │ {f'{overall_a:.4f}'.ljust(12)} │ {f'{overall_b:.4f}'.ljust(12)} │ {f'{overall_delta:+.4f}'.ljust(12)} │"
    )
    lines.append("└─────────────────────┴──────────────┴──────────────┴──────────────┘")

    if overall_delta > 0:
        lines.append(f"\n✅ {label_b} is better by {overall_delta:.4f}")
    elif overall_delta < 0:
        lines.append(f"\n❌ {label_b} is worse by {abs(overall_delta):.4f}")
    else:
        lines.append("\n➖ No difference in overall score")

    return "\n".join(lines)
