"""Tests for report metadata and enhanced report features."""

from __future__ import annotations

import json

from llm_eval.models import EvalResult, MetricResult
from llm_eval.report import (
    format_csv_report,
    format_html_report,
    format_json_report,
    format_terminal_report,
    get_report_metadata,
)


def _make_results() -> list[EvalResult]:
    """Create sample evaluation results."""
    return [
        EvalResult(
            sample_index=0,
            metrics=[
                MetricResult(
                    name="faithfulness",
                    score=0.9,
                    details={"reasoning": "Well supported"},
                ),
                MetricResult(
                    name="toxicity",
                    score=1.0,
                    details={"method": "pattern_scan", "hits": []},
                ),
            ],
        ),
        EvalResult(
            sample_index=1,
            metrics=[
                MetricResult(
                    name="faithfulness",
                    score=0.6,
                    details={"reasoning": "Partially supported"},
                ),
                MetricResult(
                    name="toxicity",
                    score=0.0,
                    details={"method": "pattern_scan", "hits": ["idiot"]},
                ),
            ],
        ),
    ]


def _make_summary() -> dict:
    """Create a sample summary."""
    return {
        "total_samples": 2,
        "overall_score": 0.625,
        "pass_count": 1,
        "fail_count": 1,
        "pass_rate": 0.5,
        "threshold": 0.7,
        "metric_scores": {
            "faithfulness": {"mean": 0.75, "min": 0.6, "max": 0.9},
            "toxicity": {"mean": 0.5, "min": 0.0, "max": 1.0},
        },
    }


def _make_metadata() -> dict:
    """Create sample metadata."""
    return {
        "timestamp": "2026-05-13T10:00:00Z",
        "version": "0.6.0",
        "python_version": "3.11.0",
        "platform": "Linux-6.1.0",
        "config_path": "evals.yaml",
        "git_hash": "abc1234",
    }


class TestGetReportMetadata:
    """Tests for get_report_metadata."""

    def test_metadata_contains_timestamp(self) -> None:
        meta = get_report_metadata()
        assert "timestamp" in meta
        assert "T" in meta["timestamp"]  # ISO format

    def test_metadata_contains_version(self) -> None:
        meta = get_report_metadata()
        assert "version" in meta
        assert meta["version"]  # not empty

    def test_metadata_contains_python_version(self) -> None:
        meta = get_report_metadata()
        assert "python_version" in meta
        assert "." in meta["python_version"]

    def test_metadata_contains_platform(self) -> None:
        meta = get_report_metadata()
        assert "platform" in meta

    def test_metadata_with_config_path(self) -> None:
        meta = get_report_metadata(config_path="evals.yaml")
        assert meta["config_path"] == "evals.yaml"


class TestTerminalReportWithMetadata:
    """Tests for terminal report with metadata."""

    def test_metadata_appears_in_report(self) -> None:
        results = _make_results()
        summary = _make_summary()
        metadata = _make_metadata()
        report = format_terminal_report(results, summary, metadata)
        assert "Version" in report
        assert "0.6.0" in report

    def test_no_metadata_still_works(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_terminal_report(results, summary)
        assert "llm-eval" in report


class TestJsonReportWithMetadata:
    """Tests for JSON report with metadata."""

    def test_metadata_in_json_output(self) -> None:
        results = _make_results()
        summary = _make_summary()
        metadata = _make_metadata()
        report = format_json_report(results, summary, metadata)
        data = json.loads(report)
        assert "metadata" in data
        assert data["metadata"]["version"] == "0.6.0"

    def test_no_metadata_key_when_absent(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_json_report(results, summary)
        data = json.loads(report)
        assert "metadata" not in data


class TestCsvReportWithMetadata:
    """Tests for CSV report with metadata."""

    def test_metadata_as_comment_lines(self) -> None:
        results = _make_results()
        metadata = _make_metadata()
        report = format_csv_report(results, metadata)
        assert "# timestamp:" in report
        assert "# version:" in report

    def test_no_metadata_no_comments(self) -> None:
        results = _make_results()
        report = format_csv_report(results)
        assert not report.startswith("#")


class TestHtmlReportWithDetails:
    """Tests for enhanced HTML report with metric details."""

    def test_contains_toggle_buttons(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_html_report(results, summary)
        assert "toggle-btn" in report
        assert "toggleDetail" in report

    def test_contains_detail_rows(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_html_report(results, summary)
        assert "detail-row" in report
        assert "detail-content" in report

    def test_contains_reasoning_in_details(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_html_report(results, summary)
        assert "Well supported" in report
        assert "Partially supported" in report

    def test_contains_pattern_hits(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_html_report(results, summary)
        assert "idiot" in report

    def test_contains_method_info(self) -> None:
        results = _make_results()
        summary = _make_summary()
        report = format_html_report(results, summary)
        assert "pattern_scan" in report

    def test_metadata_section_in_html(self) -> None:
        results = _make_results()
        summary = _make_summary()
        metadata = _make_metadata()
        report = format_html_report(results, summary, metadata)
        assert "class=\"metadata\"" in report
        assert "0.6.0" in report
