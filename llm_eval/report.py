"""Report generation for evaluation results."""

from __future__ import annotations

import csv
import io
import json
import html
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


def format_html_report(
    results: list[EvalResult], summary: dict[str, Any]
) -> str:
    """Format evaluation results as a self-contained HTML report.

    Args:
        results: List of evaluation results.
        summary: Summary statistics dictionary.

    Returns:
        Self-contained HTML string with embedded CSS.
    """
    threshold = summary.get("threshold", 0.7)
    metric_scores = summary.get("metric_scores", {})
    overall = summary.get("overall_score", 0.0)
    total = summary.get("total_samples", 0)
    pass_count = summary.get("pass_count", 0)
    fail_count = summary.get("fail_count", 0)
    overall_status = "PASS" if overall >= threshold else "FAIL"
    overall_color = "#22c55e" if overall >= threshold else "#ef4444"

    # Build metric summary rows
    metric_rows = ""
    for metric_name, scores in metric_scores.items():
        mean = scores["mean"]
        status = "PASS" if mean >= threshold else "FAIL"
        color = "#22c55e" if mean >= threshold else "#ef4444"
        metric_rows += (
            f'<tr>'
            f'<td>{html.escape(metric_name)}</td>'
            f'<td>{mean:.4f}</td>'
            f'<td>{scores["min"]:.4f}</td>'
            f'<td>{scores["max"]:.4f}</td>'
            f'<td style="color:{color};font-weight:bold">{status}</td>'
            f'</tr>\n'
        )

    # Build per-sample rows
    sample_rows = ""
    for r in results:
        metric_cells = ""
        for m in r.metrics:
            m_color = "#22c55e" if m.score >= threshold else "#ef4444"
            metric_cells += (
                f'<td style="color:{m_color}">{m.score:.4f}</td>'
            )
        row_color = "#22c55e" if r.overall_score >= threshold else "#ef4444"
        sample_rows += (
            f'<tr>'
            f'<td>#{r.sample_index}</td>'
            f'{metric_cells}'
            f'<td style="color:{row_color};font-weight:bold">{r.overall_score:.4f}</td>'
            f'</tr>\n'
        )

    # Header columns for per-sample table
    metric_header_cells = ""
    for name in metric_scores:
        metric_header_cells += f"<th>{html.escape(name)}</th>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>llm-eval Report</title>
<style>
  :root {{ --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --border: #334155; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); padding: 2rem; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
  .subtitle {{ color: #94a3b8; margin-bottom: 2rem; }}
  .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
           padding: 1.2rem 1.5rem; min-width: 140px; }}
  .card .label {{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }}
  .card .value {{ font-size: 1.6rem; font-weight: bold; margin-top: 0.3rem; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--card);
           border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-bottom: 2rem; }}
  th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ background: #0f172a; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; }}
  tr:last-child td {{ border-bottom: none; }}
  h2 {{ font-size: 1.2rem; margin-bottom: 0.8rem; }}
  .section {{ margin-bottom: 2rem; }}
  footer {{ color: #64748b; font-size: 0.8rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>🧪 llm-eval — Evaluation Report</h1>
<p class="subtitle">Generated by llm-eval</p>

<div class="cards">
  <div class="card">
    <div class="label">Overall Score</div>
    <div class="value" style="color:{overall_color}">{overall:.4f}</div>
  </div>
  <div class="card">
    <div class="label">Status</div>
    <div class="value" style="color:{overall_color}">{overall_status}</div>
  </div>
  <div class="card">
    <div class="label">Samples</div>
    <div class="value">{total}</div>
  </div>
  <div class="card">
    <div class="label">Passed</div>
    <div class="value" style="color:#22c55e">{pass_count}</div>
  </div>
  <div class="card">
    <div class="label">Failed</div>
    <div class="value" style="color:#ef4444">{fail_count}</div>
  </div>
  <div class="card">
    <div class="label">Threshold</div>
    <div class="value">{threshold:.2f}</div>
  </div>
</div>

<div class="section">
<h2>📊 Metric Summary</h2>
<table>
<tr><th>Metric</th><th>Mean</th><th>Min</th><th>Max</th><th>Status</th></tr>
{metric_rows}
</table>
</div>

<div class="section">
<h2>📋 Per-Sample Results</h2>
<table>
<tr><th>Sample</th>{metric_header_cells}<th>Overall</th></tr>
{sample_rows}
</table>
</div>

<footer>Threshold: {threshold:.2f} · {total} samples evaluated</footer>
</body>
</html>"""
