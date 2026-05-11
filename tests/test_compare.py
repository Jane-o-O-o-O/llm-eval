"""Tests for the compare module."""

from __future__ import annotations

import json

import pytest

from llm_eval.compare import (
    compare_reports,
    format_terminal_comparison,
    load_report,
)


@pytest.fixture
def report_a(tmp_path):
    """Create a sample report A."""
    data = {
        "summary": {
            "total_samples": 10,
            "overall_score": 0.82,
            "metric_scores": {
                "faithfulness": {"mean": 0.9, "min": 0.7, "max": 1.0},
                "answer_relevancy": {"mean": 0.75, "min": 0.5, "max": 0.95},
            },
        },
        "results": [],
    }
    path = tmp_path / "report_a.json"
    path.write_text(json.dumps(data))
    return str(path), data


@pytest.fixture
def report_b(tmp_path):
    """Create a sample report B."""
    data = {
        "summary": {
            "total_samples": 10,
            "overall_score": 0.88,
            "metric_scores": {
                "faithfulness": {"mean": 0.85, "min": 0.6, "max": 1.0},
                "answer_relevancy": {"mean": 0.91, "min": 0.8, "max": 1.0},
                "context_precision": {"mean": 0.78, "min": 0.5, "max": 0.95},
            },
        },
        "results": [],
    }
    path = tmp_path / "report_b.json"
    path.write_text(json.dumps(data))
    return str(path), data


class TestLoadReport:
    """Tests for load_report."""

    def test_load_valid_report(self, report_a) -> None:
        path, _ = report_a
        data = load_report(path)
        assert "summary" in data

    def test_load_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_report("/nonexistent/report.json")

    def test_load_invalid_json(self, tmp_path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("not json {{{")
        with pytest.raises(ValueError):
            load_report(str(path))

    def test_load_missing_summary(self, tmp_path) -> None:
        path = tmp_path / "no_summary.json"
        path.write_text(json.dumps({"results": []}))
        with pytest.raises(ValueError):
            load_report(str(path))


class TestCompareReports:
    """Tests for compare_reports."""

    def test_compare_basic(self, report_a, report_b) -> None:
        path_a, data_a = report_a
        path_b, data_b = report_b
        result = compare_reports(data_a, data_b, "v1", "v2")

        assert result["overall_delta"] == pytest.approx(0.06, abs=0.01)
        assert len(result["comparisons"]) == 3  # 3 unique metrics

    def test_compare_preserves_labels(self, report_a, report_b) -> None:
        path_a, data_a = report_a
        path_b, data_b = report_b
        result = compare_reports(data_a, data_b, "Baseline", "Experiment")

        assert "Baseline" in result
        assert "Experiment" in result

    def test_compare_metric_deltas(self, report_a, report_b) -> None:
        path_a, data_a = report_a
        path_b, data_b = report_b
        result = compare_reports(data_a, data_b, "A", "B")

        # faithfulness: 0.85 - 0.90 = -0.05
        faith_comp = next(c for c in result["comparisons"] if c["metric"] == "faithfulness")
        assert faith_comp["delta"] == pytest.approx(-0.05, abs=0.01)

        # answer_relevancy: 0.91 - 0.75 = 0.16
        rel_comp = next(c for c in result["comparisons"] if c["metric"] == "answer_relevancy")
        assert rel_comp["delta"] == pytest.approx(0.16, abs=0.01)

    def test_compare_missing_metric_in_a(self, report_b) -> None:
        path_b, data_b = report_b
        data_a = {
            "summary": {
                "overall_score": 0.8,
                "metric_scores": {"faithfulness": {"mean": 0.8}},
            }
        }
        result = compare_reports(data_a, data_b, "A", "B")
        # context_precision only in B, should default to 0 for A
        ctx_comp = next(c for c in result["comparisons"] if c["metric"] == "context_precision")
        assert ctx_comp["A"] == 0.0
        assert ctx_comp["B"] == pytest.approx(0.78, abs=0.01)


class TestFormatTerminalComparison:
    """Tests for format_terminal_comparison."""

    def test_contains_metric_names(self, report_a, report_b) -> None:
        path_a, data_a = report_a
        path_b, data_b = report_b
        comp = compare_reports(data_a, data_b, "v1", "v2")
        output = format_terminal_comparison(comp)
        assert "faithfulness" in output
        assert "answer_relevancy" in output

    def test_contains_labels(self, report_a, report_b) -> None:
        path_a, data_a = report_a
        path_b, data_b = report_b
        comp = compare_reports(data_a, data_b, "Baseline", "Current")
        output = format_terminal_comparison(comp)
        assert "Baseline" in output
        assert "Current" in output

    def test_shows_improvement(self, report_a, report_b) -> None:
        path_a, data_a = report_a
        path_b, data_b = report_b
        comp = compare_reports(data_a, data_b, "A", "B")
        output = format_terminal_comparison(comp)
        # overall_delta is positive (B better than A)
        assert "better" in output.lower() or "✅" in output

    def test_shows_regression(self, report_a, report_b) -> None:
        path_a, data_a = report_a
        path_b, data_b = report_b
        comp = compare_reports(data_b, data_a, "B", "A")  # A is worse
        output = format_terminal_comparison(comp)
        assert "worse" in output.lower() or "❌" in output
