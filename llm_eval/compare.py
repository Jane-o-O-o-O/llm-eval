"""Compare evaluation results across runs or models."""

from __future__ import annotations

import html
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
    with open(path, encoding="utf-8") as f:
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
        score_a = (
            metrics_a.get(metric, {}).get("mean", 0.0)
            if isinstance(metrics_a.get(metric), dict)
            else float(metrics_a.get(metric, 0.0))
        )
        score_b = (
            metrics_b.get(metric, {}).get("mean", 0.0)
            if isinstance(metrics_b.get(metric), dict)
            else float(metrics_b.get(metric, 0.0))
        )
        delta = score_b - score_a

        comparisons.append(
            {
                "metric": metric,
                label_a: round(score_a, 4),
                label_b: round(score_b, 4),
                "delta": round(delta, 4),
            }
        )

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


def format_html_comparison(comparison: dict[str, Any]) -> str:
    """Format a comparison as a self-contained HTML report.

    Args:
        comparison: Comparison dictionary from compare_reports.

    Returns:
        Self-contained HTML string with embedded CSS and side-by-side chart.
    """
    labels = [k for k in comparison if k not in ("overall_delta", "comparisons")]
    label_a = labels[0] if len(labels) > 0 else "Report A"
    label_b = labels[1] if len(labels) > 1 else "Report B"

    overall_a = comparison[label_a]["overall"]
    overall_b = comparison[label_b]["overall"]
    overall_delta = comparison["overall_delta"]

    if overall_delta > 0:
        verdict = f"✅ {label_b} is better by {overall_delta:.4f}"
        verdict_color = "#22c55e"
    elif overall_delta < 0:
        verdict = f"❌ {label_b} is worse by {abs(overall_delta):.4f}"
        verdict_color = "#ef4444"
    else:
        verdict = "➖ No difference in overall score"
        verdict_color = "#94a3b8"

    # Build comparison rows
    rows = ""
    for c in comparison.get("comparisons", []):
        metric = c["metric"]
        val_a = c[label_a]
        val_b = c[label_b]
        delta = c["delta"]

        if delta > 0.001:
            delta_color = "#22c55e"
            delta_icon = "▲"
        elif delta < -0.001:
            delta_color = "#ef4444"
            delta_icon = "▼"
        else:
            delta_color = "#94a3b8"
            delta_icon = "─"

        rows += (
            f"<tr>"
            f"<td>{html.escape(metric)}</td>"
            f"<td>{val_a:.4f}</td>"
            f"<td>{val_b:.4f}</td>"
            f'<td style="color:{delta_color};font-weight:bold">{delta_icon} {delta:+.4f}</td>'
            f"</tr>\n"
        )

    # Build SVG side-by-side bar chart
    metrics = comparison.get("comparisons", [])
    bar_height = 28
    chart_height = len(metrics) * (bar_height * 2 + 20) + 40
    chart_width = 600
    max_val = 1.0  # scores are 0-1

    svg_bars = ""
    for i, c in enumerate(metrics):
        y_base = 20 + i * (bar_height * 2 + 20)
        name = c["metric"]
        val_a = c[label_a]
        val_b = c[label_b]

        bar_a_width = (val_a / max_val) * (chart_width - 160) if max_val > 0 else 0
        bar_b_width = (val_b / max_val) * (chart_width - 160) if max_val > 0 else 0

        # Label
        svg_bars += f'<text x="5" y="{y_base + bar_height + 4}" fill="#e2e8f0" font-size="11" font-weight="bold">{html.escape(name)}</text>'

        # Bar A
        svg_bars += f'<rect x="120" y="{y_base}" width="{bar_a_width}" height="{bar_height - 2}" fill="#60a5fa" rx="3" opacity="0.8"/>'
        svg_bars += f'<text x="{125 + bar_a_width}" y="{y_base + bar_height - 6}" fill="#94a3b8" font-size="10">{val_a:.3f}</text>'

        # Bar B
        svg_bars += f'<rect x="120" y="{y_base + bar_height}" width="{bar_b_width}" height="{bar_height - 2}" fill="#a78bfa" rx="3" opacity="0.8"/>'
        svg_bars += f'<text x="{125 + bar_b_width}" y="{y_base + bar_height * 2 - 6}" fill="#94a3b8" font-size="10">{val_b:.3f}</text>'

    chart_svg = (
        f'<svg width="{chart_width}" height="{chart_height}" xmlns="http://www.w3.org/2000/svg">'
        f'<text x="120" y="14" fill="#60a5fa" font-size="11">{html.escape(label_a)}</text>'
        f'<text x="220" y="14" fill="#a78bfa" font-size="11">{html.escape(label_b)}</text>'
        f"{svg_bars}</svg>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>llm-eval — Comparison</title>
<style>
  :root {{ --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --border: #334155; }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: var(--bg); color: var(--text); padding: 2rem; }}
  h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
  .subtitle {{ color: #94a3b8; margin-bottom: 2rem; }}
  .verdict {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
              padding: 1rem 1.5rem; margin-bottom: 2rem; font-size: 1.2rem; font-weight: bold; }}
  .cards {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 2rem; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
           padding: 1.2rem 1.5rem; min-width: 160px; }}
  .card .label {{ font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; }}
  .card .value {{ font-size: 1.4rem; font-weight: bold; margin-top: 0.3rem; }}
  table {{ width: 100%; border-collapse: collapse; background: var(--card);
           border: 1px solid var(--border); border-radius: 8px; overflow: hidden; margin-bottom: 2rem; }}
  th, td {{ padding: 0.6rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
  th {{ background: #0f172a; color: #94a3b8; font-size: 0.8rem; text-transform: uppercase; }}
  tr:last-child td {{ border-bottom: none; }}
  h2 {{ font-size: 1.2rem; margin-bottom: 0.8rem; }}
  .section {{ margin-bottom: 2rem; }}
  svg {{ display: block; margin: 1rem 0; background: var(--card); border-radius: 8px; padding: 0.5rem; }}
  footer {{ color: #64748b; font-size: 0.8rem; margin-top: 2rem; }}
</style>
</head>
<body>
<h1>📊 llm-eval — Comparison Report</h1>
<p class="subtitle">{html.escape(label_a)} vs {html.escape(label_b)}</p>

<div class="cards">
  <div class="card">
    <div class="label">{html.escape(label_a)} Overall</div>
    <div class="value" style="color:#60a5fa">{overall_a:.4f}</div>
  </div>
  <div class="card">
    <div class="label">{html.escape(label_b)} Overall</div>
    <div class="value" style="color:#a78bfa">{overall_b:.4f}</div>
  </div>
  <div class="card">
    <div class="label">Delta</div>
    <div class="value" style="color:{verdict_color}">{overall_delta:+.4f}</div>
  </div>
</div>

<div class="verdict" style="color:{verdict_color};border-color:{verdict_color}40">
  {verdict}
</div>

<div class="section">
<h2>📊 Per-Metric Comparison</h2>
<table>
<tr><th>Metric</th><th>{html.escape(label_a)}</th><th>{html.escape(label_b)}</th><th>Delta</th></tr>
{rows}
</table>
</div>

<div class="section">
<h2>📈 Visual Comparison</h2>
{chart_svg}
</div>

<footer>llm-eval comparison report</footer>
</body>
</html>"""

# [2026-05-13] batch evaluation
class BatchEvaluationHandler:
    """Handler for batch evaluation operations."""

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._initialized = False
        self._cache = {}

    def initialize(self) -> bool:
        """Initialize the handler with current configuration."""
        if self._initialized:
            return True
        try:
            self._validate_config()
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"Initialization failed: {e}")
            return False

    def _validate_config(self):
        """Validate configuration parameters."""
        required = self._required_keys()
        missing = [k for k in required if k not in self._config]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")

    def _required_keys(self) -> list:
        return ["enabled"]

    def process(self, data: dict) -> dict:
        """Process data through the handler."""
        if not self._initialized:
            self.initialize()
        result = self._transform(data)
        self._cache[data.get("id", "default")] = result
        return result

    def _transform(self, data: dict) -> dict:
        """Apply transformation to input data."""
        return {"status": "processed", "data": data, "handler": self.__class__.__name__}

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()

# [2026-05-25] Documentation update for compare
"""
Compare Module

This module provides CLI progress bar functionality.

Usage:
    from llm_eval.compare import process

    result = process(data, config={"enabled": True})

Configuration:
    - enabled (bool): Enable/disable the module. Default: True
    - debug (bool): Enable debug logging. Default: False
    - timeout (int): Operation timeout in seconds. Default: 30

Added: 2026-05-25
"""
