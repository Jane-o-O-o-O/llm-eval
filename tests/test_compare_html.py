"""Tests for HTML comparison report."""

from __future__ import annotations

import json

from llm_eval.compare import (
    compare_reports,
    format_html_comparison,
    format_terminal_comparison,
)


def _make_comparison() -> dict:
    """Create a sample comparison dictionary."""
    report_a = {
        "summary": {
            "total_samples": 10,
            "overall_score": 0.82,
            "metric_scores": {
                "faithfulness": {"mean": 0.9, "min": 0.7, "max": 1.0},
                "answer_relevancy": {"mean": 0.75, "min": 0.5, "max": 0.95},
            },
        },
    }
    report_b = {
        "summary": {
            "total_samples": 10,
            "overall_score": 0.88,
            "metric_scores": {
                "faithfulness": {"mean": 0.85, "min": 0.6, "max": 1.0},
                "answer_relevancy": {"mean": 0.91, "min": 0.8, "max": 1.0},
            },
        },
    }
    return compare_reports(report_a, report_b, "Baseline", "Current")


class TestHtmlComparison:
    """Tests for format_html_comparison."""

    def test_valid_html_output(self) -> None:
        comparison = _make_comparison()
        html_output = format_html_comparison(comparison)
        assert "<!DOCTYPE html>" in html_output
        assert "</html>" in html_output

    def test_contains_labels(self) -> None:
        comparison = _make_comparison()
        html_output = format_html_comparison(comparison)
        assert "Baseline" in html_output
        assert "Current" in html_output

    def test_contains_metric_names(self) -> None:
        comparison = _make_comparison()
        html_output = format_html_comparison(comparison)
        assert "faithfulness" in html_output
        assert "answer_relevancy" in html_output

    def test_contains_overall_scores(self) -> None:
        comparison = _make_comparison()
        html_output = format_html_comparison(comparison)
        assert "0.82" in html_output
        assert "0.88" in html_output

    def test_contains_svg_chart(self) -> None:
        comparison = _make_comparison()
        html_output = format_html_comparison(comparison)
        assert "<svg" in html_output
        assert "</svg>" in html_output

    def test_contains_comparison_table(self) -> None:
        comparison = _make_comparison()
        html_output = format_html_comparison(comparison)
        assert "Per-Metric Comparison" in html_output
        assert "<table>" in html_output

    def test_contains_verdict(self) -> None:
        comparison = _make_comparison()
        html_output = format_html_comparison(comparison)
        assert "better" in html_output.lower()

    def test_contains_css_styling(self) -> None:
        comparison = _make_comparison()
        html_output = format_html_comparison(comparison)
        assert "<style>" in html_output
        assert "background" in html_output

    def test_regression_verdict(self) -> None:
        """Test with reversed scores to show regression."""
        report_a = {
            "summary": {
                "overall_score": 0.90,
                "metric_scores": {
                    "faithfulness": {"mean": 0.95, "min": 0.9, "max": 1.0},
                },
            },
        }
        report_b = {
            "summary": {
                "overall_score": 0.70,
                "metric_scores": {
                    "faithfulness": {"mean": 0.70, "min": 0.5, "max": 0.9},
                },
            },
        }
        comparison = compare_reports(report_a, report_b, "v1", "v2")
        html_output = format_html_comparison(comparison)
        assert "worse" in html_output.lower()

    def test_delta_arrows(self) -> None:
        """Test that delta arrows appear in the output."""
        comparison = _make_comparison()
        html_output = format_html_comparison(comparison)
        # Delta arrows: ▲, ▼, or ─
        assert "▲" in html_output or "▼" in html_output or "─" in html_output
