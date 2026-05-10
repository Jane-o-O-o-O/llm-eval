"""Report generation for evaluation results."""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from llm_eval.models import EvalResult


def format_terminal_report(
    results: list[EvalResult], summary: dict[str, Any]
) -> str:
    """Format evaluation results as a terminal-friendly table.

    Args:
        results: List of evaluation results.
        summary: Summary statistics dictionary.

    Returns:
        Formatted string for terminal display.
    """
    threshold = summary.get("threshold", 0.7)
    lines: list[str] = []
    lines.append("┌─────────────────────────────────────────────────┐")
    lines.append("│          🧪 llm-eval — Evaluation Report         │")
    lines.append("├─────────────────────┬───────────┬───────────────┤")
    lines.append("│ Metric              │ Score     │ Status        │")
    lines.append("├─────────────────────┼───────────┼───────────────┤")

    metric_scores = summary.get("metric_scores", {})
    for metric_name, scores in metric_scores.items():
        mean = scores["mean"]
        status = "✅ PASS" if mean >= threshold else "❌ FAIL"
        name_col = metric_name.ljust(19)
        score_col = f"{mean:.2f}".ljust(9)
        lines.append(f"│ {name_col} │ {score_col} │ {status.ljust(13)} │")

    lines.append("├─────────────────────┼───────────┼───────────────┤")
    overall = summary["overall_score"]
    overall_status = "✅ PASS" if overall >= threshold else "❌ FAIL"
    lines.append(
        f"│ {'Overall'.ljust(19)} │ {f'{overall:.2f}'.ljust(9)} │ {overall_status.ljust(13)} │"
    )
    lines.append("└─────────────────────┴───────────┴───────────────┘")
    lines.append(
        f"Evaluated {summary['total_samples']} samples "
        f"({summary['pass_count']} passed, {summary['fail_count']} failed)"
    )
    return "\n".join(lines)


def format_json_report(
    results: list[EvalResult], summary: dict[str, Any]
) -> str:
    """Format evaluation results as JSON.

    Args:
        results: List of evaluation results.
        summary: Summary statistics dictionary.

    Returns:
        JSON string of the report.
    """
    report = {
        "summary": summary,
        "results": [r.to_dict() for r in results],
    }
    return json.dumps(report, indent=2, ensure_ascii=False)


def format_csv_report(results: list[EvalResult]) -> str:
    """Format evaluation results as CSV.

    Args:
        results: List of evaluation results.

    Returns:
        CSV string of the report.
    """
    if not results:
        return ""

    # Collect all metric names
    metric_names: list[str] = []
    for r in results:
        for m in r.metrics:
            if m.name not in metric_names:
                metric_names.append(m.name)

    output = io.StringIO()
    writer = csv.writer(output)
    header = ["sample_index", "overall_score"] + metric_names
    writer.writerow(header)

    for result in results:
        metric_map = {m.name: m.score for m in result.metrics}
        row = [result.sample_index, f"{result.overall_score:.4f}"]
        row.extend(f"{metric_map.get(name, 0.0):.4f}" for name in metric_names)
        writer.writerow(row)

    return output.getvalue()
