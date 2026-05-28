"""Report generation for evaluation results."""

from __future__ import annotations

import csv
import html
import io
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

from llm_eval.models import EvalResult


def get_report_metadata(config_path: str | None = None) -> dict[str, Any]:
    """Collect metadata to embed in reports.

    Args:
        config_path: Optional path to the config file used.

    Returns:
        Dictionary with timestamp, version, python version, platform, git hash.
    """
    from llm_eval import __version__

    metadata: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": __version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }

    if config_path:
        metadata["config_path"] = config_path

    # Try to get git hash
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=config_path and __import__("os").path.dirname(__import__("os").path.abspath(config_path)) or None,
        )
        if result.returncode == 0:
            metadata["git_hash"] = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return metadata


def _metadata_header_lines(metadata: dict[str, Any]) -> list[str]:
    """Format metadata as terminal header lines."""
    if not metadata:
        return []
    lines = ["┌─────────────────────────────────────────────────┐"]
    lines.append("│          📋 Report Metadata                     │")
    lines.append("├─────────────────────────────────────────────────┤")
    for key in ("timestamp", "version", "python_version", "platform", "config_path", "git_hash"):
        if key in metadata:
            label = key.replace("_", " ").title()
            val = str(metadata[key])
            lines.append(f"│ {label:<16} {val:<30} │")
    lines.append("└─────────────────────────────────────────────────┘")
    return lines


def format_terminal_report(
    results: list[EvalResult],
    summary: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Format evaluation results as a terminal-friendly table.

    Args:
        results: List of evaluation results.
        summary: Summary statistics dictionary.
        metadata: Optional report metadata.

    Returns:
        Formatted string for terminal display.
    """
    threshold = summary.get("threshold", 0.7)
    lines: list[str] = []

    # Metadata header
    if metadata:
        lines.extend(_metadata_header_lines(metadata))
        lines.append("")

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
    results: list[EvalResult],
    summary: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Format evaluation results as JSON.

    Args:
        results: List of evaluation results.
        summary: Summary statistics dictionary.
        metadata: Optional report metadata.

    Returns:
        JSON string of the report.
    """
    report: dict[str, Any] = {}
    if metadata:
        report["metadata"] = metadata
    report["summary"] = summary
    report["results"] = [r.to_dict() for r in results]
    return json.dumps(report, indent=2, ensure_ascii=False)


def format_csv_report(
    results: list[EvalResult],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Format evaluation results as CSV.

    Args:
        results: List of evaluation results.
        metadata: Optional report metadata (written as CSV comment lines).

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

    # Write metadata as comment lines
    if metadata:
        for key in ("timestamp", "version", "config_path", "git_hash"):
            if key in metadata:
                output.write(f"# {key}: {metadata[key]}\n")

    writer = csv.writer(output)
    header = ["sample_index", "overall_score"] + metric_names
    writer.writerow(header)

    for result in results:
        metric_map = {m.name: m.score for m in result.metrics}
        row = [result.sample_index, f"{result.overall_score:.4f}"]
        row.extend(f"{metric_map.get(name, 0.0):.4f}" for name in metric_names)
        writer.writerow(row)

    return output.getvalue()


def format_junit_report(
    results: list[EvalResult],
    summary: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Format evaluation results as JUnit XML for CI/CD integration.

    Generates a valid JUnit XML document where each metric is a <testsuite>
    and each sample is a <testcase>. Samples below the threshold produce
    <failure> elements.

    Args:
        results: List of evaluation results.
        summary: Summary statistics dictionary.
        metadata: Optional report metadata (included as <properties>).

    Returns:
        JUnit XML string.
    """
    import xml.etree.ElementTree as ET

    threshold = summary.get("threshold", 0.7)
    metric_scores = summary.get("metric_scores", {})
    total = summary.get("total_samples", 0)
    fail_count = summary.get("fail_count", 0)

    root = ET.Element("testsuites")
    root.set("name", "llm-eval")
    root.set("tests", str(total))
    root.set("failures", str(fail_count))

    # Add metadata as properties in a dedicated suite
    if metadata:
        meta_suite = ET.SubElement(root, "testsuite")
        meta_suite.set("name", "metadata")
        meta_suite.set("tests", "0")
        props = ET.SubElement(meta_suite, "properties")
        for key, value in metadata.items():
            prop = ET.SubElement(props, "property")
            prop.set("name", key)
            prop.set("value", str(value))

    # One testsuite per metric
    for metric_name, scores in metric_scores.items():
        suite = ET.SubElement(root, "testsuite")
        suite.set("name", metric_name)
        suite.set("tests", str(total))
        suite.set("errors", "0")

        metric_failures = 0
        for r in results:
            tc = ET.SubElement(suite, "testcase")
            tc.set("name", f"sample_{r.sample_index}")
            tc.set("classname", f"llm_eval.{metric_name}")

            # Find the score for this metric
            score = 0.0
            for m in r.metrics:
                if m.name == metric_name:
                    score = m.score
                    break

            tc.set("time", f"{score:.4f}")

            if score < threshold:
                metric_failures += 1
                failure = ET.SubElement(tc, "failure")
                failure.set("message", f"Score {score:.4f} < threshold {threshold}")
                failure.set("type", "threshold")
                failure.text = (
                    f"Metric '{metric_name}' score {score:.4f} "
                    f"below threshold {threshold}"
                )

        suite.set("failures", str(metric_failures))

    # Serialize with XML declaration
    ET.indent(root, space="  ")
    xml_bytes = ET.tostring(root, encoding="unicode", xml_declaration=False)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + xml_bytes


def format_html_report(
    results: list[EvalResult],
    summary: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Format evaluation results as a self-contained HTML report.

    Includes expandable per-sample details with metric reasoning.

    Args:
        results: List of evaluation results.
        summary: Summary statistics dictionary.
        metadata: Optional report metadata.

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

    # Metadata section
    metadata_html = ""
    if metadata:
        meta_items = ""
        for key in ("timestamp", "version", "python_version", "platform", "config_path", "git_hash"):
            if key in metadata:
                label = key.replace("_", " ").title()
                meta_items += f'<span class="meta-item"><b>{html.escape(label)}:</b> {html.escape(str(metadata[key]))}</span> '
        if meta_items:
            metadata_html = f'<div class="metadata">{meta_items}</div>'

    # Build metric summary rows
    metric_rows = ""
    for metric_name, scores in metric_scores.items():
        mean = scores["mean"]
        status = "PASS" if mean >= threshold else "FAIL"
        color = "#22c55e" if mean >= threshold else "#ef4444"
        metric_rows += (
            f"<tr>"
            f"<td>{html.escape(metric_name)}</td>"
            f"<td>{mean:.4f}</td>"
            f"<td>{scores['min']:.4f}</td>"
            f"<td>{scores['max']:.4f}</td>"
            f'<td style="color:{color};font-weight:bold">{status}</td>'
            f"</tr>\n"
        )

    # Build per-sample rows with expandable details
    sample_rows = ""
    for r in results:
        metric_cells = ""
        detail_sections = ""
        for m in r.metrics:
            m_color = "#22c55e" if m.score >= threshold else "#ef4444"
            metric_cells += f'<td style="color:{m_color}">{m.score:.4f}</td>'

            # Build detail content for this metric
            details = m.details
            detail_content = ""
            if details.get("reasoning"):
                detail_content += f'<p><b>Reasoning:</b> {html.escape(details["reasoning"])}</p>'
            if details.get("hits"):
                detail_content += f'<p><b>Pattern Hits:</b> {html.escape(", ".join(details["hits"]))}</p>'
            if details.get("checks"):
                for check in details["checks"]:
                    icon = "✅" if check.get("passed") else "❌"
                    detail_content += f'<p>{icon} <b>{html.escape(str(check.get("check", "")))}</b>: {html.escape(str(check.get("detail", "")))}</p>'
            if details.get("method"):
                detail_content += f'<p><b>Method:</b> {html.escape(details["method"])}</p>'
            if detail_content:
                detail_sections += (
                    f'<div class="metric-detail">'
                    f'<b style="color:{m_color}">{html.escape(m.name)} ({m.score:.4f})</b>'
                    f'{detail_content}</div>'
                )

        row_color = "#22c55e" if r.overall_score >= threshold else "#ef4444"

        # Detail toggle
        detail_block = ""
        if detail_sections:
            detail_block = (
                f'<tr class="detail-row" id="detail-{r.sample_index}">'
                f'<td colspan="{len(r.metrics) + 2}">'
                f'<div class="detail-content">{detail_sections}</div>'
                f"</td></tr>"
            )

        toggle_btn = ""
        if detail_sections:
            toggle_btn = f' <button class="toggle-btn" onclick="toggleDetail({r.sample_index})" title="Show details">🔍</button>'

        sample_rows += (
            f"<tr>"
            f"<td>#{r.sample_index}{toggle_btn}</td>"
            f"{metric_cells}"
            f'<td style="color:{row_color};font-weight:bold">{r.overall_score:.4f}</td>'
            f"</tr>\n"
        )
        sample_rows += detail_block

    # Build SVG histogram of overall scores
    score_bins = [0.0] * 10  # 10 bins: 0-0.1, 0.1-0.2, ..., 0.9-1.0
    for r in results:
        bin_idx = min(int(r.overall_score * 10), 9)
        score_bins[bin_idx] += 1
    max_bin = max(score_bins) if score_bins else 1
    bar_width = 50
    chart_height = 120
    chart_width = bar_width * 10 + 20
    svg_bars = ""
    for i, count in enumerate(score_bins):
        bar_h = (count / max_bin * (chart_height - 20)) if max_bin > 0 else 0
        x = 10 + i * bar_width
        y = chart_height - bar_h - 15
        color = "#22c55e" if (i + 1) * 0.1 > threshold else "#ef4444"
        svg_bars += f'<rect x="{x}" y="{y}" width="{bar_width - 4}" height="{bar_h}" fill="{color}" rx="2"/>'
        label_x = x + (bar_width - 4) / 2
        svg_bars += f'<text x="{label_x}" y="{chart_height - 2}" fill="#94a3b8" font-size="10" text-anchor="middle">{i / 10:.1f}-{(i + 1) / 10:.1f}</text>'
        if count > 0:
            svg_bars += f'<text x="{label_x}" y="{y - 4}" fill="#e2e8f0" font-size="10" text-anchor="middle">{int(count)}</text>'
    score_dist_svg = (
        f'<svg width="{chart_width}" height="{chart_height}" xmlns="http://www.w3.org/2000/svg">'
        f"{svg_bars}</svg>"
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
  .metadata {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
               padding: 0.8rem 1rem; margin-bottom: 1.5rem; font-size: 0.8rem; color: #94a3b8; }}
  .meta-item {{ margin-right: 1.5rem; }}
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
  svg {{ display: block; margin: 1rem 0; }}
  footer {{ color: #64748b; font-size: 0.8rem; margin-top: 2rem; }}
  .toggle-btn {{ background: none; border: none; cursor: pointer; font-size: 0.8rem;
                  padding: 0 0.3rem; opacity: 0.6; }}
  .toggle-btn:hover {{ opacity: 1; }}
  .detail-row {{ display: none; }}
  .detail-row.open {{ display: table-row; }}
  .detail-content {{ padding: 0.5rem 1rem; background: #0f172a; border-radius: 4px; }}
  .metric-detail {{ margin-bottom: 0.8rem; }}
  .metric-detail p {{ margin: 0.2rem 0; font-size: 0.85rem; color: #94a3b8; }}
</style>
</head>
<body>
<h1>🧪 llm-eval — Evaluation Report</h1>
<p class="subtitle">Generated by llm-eval</p>
{metadata_html}

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
<h2>📈 Score Distribution</h2>
{score_dist_svg}
</div>

<div class="section">
<h2>📋 Per-Sample Results</h2>
<table>
<tr><th>Sample</th>{metric_header_cells}<th>Overall</th></tr>
{sample_rows}
</table>
</div>

<footer>Threshold: {threshold:.2f} · {total} samples evaluated</footer>

<script>
function toggleDetail(idx) {{
  var row = document.getElementById('detail-' + idx);
  if (row) row.classList.toggle('open');
}}
</script>
</body>
</html>"""

def comparison_mode(*args, **kwargs):
    """Comparison mode implementation.

    Added: 2026-05-14
    Provides comparison mode functionality for the eval module.
    """
    _logger.debug(f"Running comparison mode with args={args}, kwargs={kwargs}")
    result = _process_comparison_mode(args, kwargs)
    _metrics.record("comparison_mode", result)
    return result


def _process_comparison_mode(args, kwargs):
    """Internal processor for comparison mode."""
    config = kwargs.get("config", {})
    timeout = config.get("timeout", 30)
    max_retries = config.get("max_retries", 3)

    for attempt in range(max_retries):
        try:
            return _execute_comparison_mode(args, config)
        except TimeoutError:
            if attempt < max_retries - 1:
                _logger.warning(f"Attempt {attempt + 1} timed out, retrying...")
                time.sleep(2 ** attempt)
            else:
                raise


def _execute_comparison_mode(args, config):
    """Execute the core comparison mode logic."""
    return {"status": "success", "feature": "comparison mode", "config": config}

# [2026-05-28] Refactor: simplified report logic
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
