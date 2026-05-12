"""Tests for report generation including HTML format."""

import json

import pytest
from llm_eval.models import EvalResult, MetricResult
from llm_eval.report import (
    format_csv_report,
    format_html_report,
    format_json_report,
    format_terminal_report,
)


def _make_results() -> list[EvalResult]:
    """Create sample evaluation results for testing."""
    return [
        EvalResult(
            sample_index=0,
            metrics=[
                MetricResult(name="faithfulness", score=0.9),
                MetricResult(name="answer_relevancy", score=0.8),
            ],
        ),
        EvalResult(
            sample_index=1,
            metrics=[
                MetricResult(name="faithfulness", score=0.6),
                MetricResult(name="answer_relevancy", score=0.7),
            ],
        ),
    ]


def _make_summary() -> dict:
    """Create a sample summary for testing."""
    return {
        "total_samples": 2,
        "overall_score": 0.75,
        "pass_count": 1,
        "fail_count": 1,
        "pass_rate": 0.5,
        "threshold": 0.7,
        "metric_scores": {
            "faithfulness": {"mean": 0.75, "min": 0.6, "max": 0.9},
            "answer_relevancy": {"mean": 0.75, "min": 0.7, "max": 0.8},
        },
    }


class TestTerminalReport:
    def test_contains_table(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_terminal_report(results, summary)
        assert "llm-eval" in report
        assert "faithfulness" in report
        assert "answer_relevancy" in report

    def test_shows_pass_fail(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_terminal_report(results, summary)
        assert "PASS" in report or "FAIL" in report


class TestJsonReport:
    def test_valid_json(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_json_report(results, summary)
        data = json.loads(report)
        assert "summary" in data
        assert "results" in data
        assert len(data["results"]) == 2

    def test_summary_fields(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_json_report(results, summary)
        data = json.loads(report)
        assert data["summary"]["total_samples"] == 2
        assert data["summary"]["overall_score"] == 0.75


class TestCsvReport:
    def test_csv_header(self) -> None:
        results = _make_results()
        report = format_csv_report(results)
        lines = report.strip().split("\n")
        assert "sample_index" in lines[0]
        assert "faithfulness" in lines[0]

    def test_csv_data_rows(self) -> None:
        results = _make_results()
        report = format_csv_report(results)
        lines = report.strip().split("\n")
        assert len(lines) == 3  # header + 2 data rows

    def test_empty_results(self) -> None:
        assert format_csv_report([]) == ""


class TestHtmlReport:
    def test_valid_html(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_html_report(results, summary)
        assert "<!DOCTYPE html>" in report
        assert "</html>" in report

    def test_contains_summary_cards(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_html_report(results, summary)
        assert "Overall Score" in report
        assert "Passed" in report
        assert "Failed" in report
        assert "Threshold" in report

    def test_contains_metric_table(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_html_report(results, summary)
        assert "faithfulness" in report
        assert "answer_relevancy" in report
        assert "Metric Summary" in report

    def test_contains_per_sample_table(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_html_report(results, summary)
        assert "Per-Sample Results" in report
        assert "#0" in report
        assert "#1" in report

    def test_contains_threshold(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_html_report(results, summary)
        assert "0.70" in report

    def test_css_styling(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_html_report(results, summary)
        assert "<style>" in report
        assert "background" in report

    def test_empty_results(self) -> None:
        empty_summary = {
            "total_samples": 0,
            "overall_score": 0.0,
            "pass_count": 0,
            "fail_count": 0,
            "pass_rate": 0.0,
            "threshold": 0.7,
            "metric_scores": {},
        }
        report = format_html_report([], empty_summary)
        assert "<!DOCTYPE html>" in report
        assert "0 samples evaluated" in report
