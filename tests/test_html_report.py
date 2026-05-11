"""Tests for the HTML report generation."""

import json

import pytest
from llm_eval.report import format_html_report
from llm_eval.models import EvalResult, MetricResult


class TestHtmlReport:
    """Tests for the HTML report format."""

    def _make_results_and_summary(self):
        results = [
            EvalResult(
                sample_index=0,
                metrics=[
                    MetricResult(name="faithfulness", score=0.95, details={"reasoning": "Good"}),
                    MetricResult(name="answer_relevancy", score=0.85, details={}),
                ],
            ),
            EvalResult(
                sample_index=1,
                metrics=[
                    MetricResult(name="faithfulness", score=0.60, details={"reasoning": "Weak"}),
                    MetricResult(name="answer_relevancy", score=0.40, details={}),
                ],
            ),
        ]
        summary = {
            "total_samples": 2,
            "overall_score": 0.70,
            "pass_count": 1,
            "fail_count": 1,
            "pass_rate": 0.5,
            "threshold": 0.7,
            "metric_scores": {
                "faithfulness": {"mean": 0.775, "min": 0.60, "max": 0.95},
                "answer_relevancy": {"mean": 0.625, "min": 0.40, "max": 0.85},
            },
        }
        return results, summary

    def test_html_is_valid_string(self) -> None:
        results, summary = self._make_results_and_summary()
        html = format_html_report(results, summary)
        assert isinstance(html, str)

    def test_html_contains_doctype(self) -> None:
        results, summary = self._make_results_and_summary()
        html = format_html_report(results, summary)
        assert "<!DOCTYPE html>" in html

    def test_html_contains_title(self) -> None:
        results, summary = self._make_results_and_summary()
        html = format_html_report(results, summary)
        assert "llm-eval" in html.lower()

    def test_html_contains_metric_names(self) -> None:
        results, summary = self._make_results_and_summary()
        html = format_html_report(results, summary)
        assert "faithfulness" in html
        assert "answer_relevancy" in html

    def test_html_contains_scores(self) -> None:
        results, summary = self._make_results_and_summary()
        html = format_html_report(results, summary)
        assert "0.95" in html
        assert "0.60" in html

    def test_html_contains_pass_fail_info(self) -> None:
        results, summary = self._make_results_and_summary()
        html = format_html_report(results, summary)
        assert "PASS" in html or "pass" in html
        assert "FAIL" in html or "fail" in html

    def test_html_contains_summary_stats(self) -> None:
        results, summary = self._make_results_and_summary()
        html = format_html_report(results, summary)
        assert "0.70" in html or "0.7" in html
        assert "2" in html  # total_samples

    def test_html_is_self_contained_css(self) -> None:
        """HTML should include embedded CSS, not external stylesheets."""
        results, summary = self._make_results_and_summary()
        html = format_html_report(results, summary)
        assert "<style>" in html

    def test_html_empty_results(self) -> None:
        results = []
        summary = {
            "total_samples": 0,
            "overall_score": 0.0,
            "pass_count": 0,
            "fail_count": 0,
            "pass_rate": 0.0,
            "threshold": 0.7,
            "metric_scores": {},
        }
        html = format_html_report(results, summary)
        assert "<!DOCTYPE html>" in html
        assert "0" in html

    def test_html_contains_sample_details(self) -> None:
        results, summary = self._make_results_and_summary()
        html = format_html_report(results, summary)
        # Should have per-sample info
        assert "0.95" in html  # sample 0 faithfulness
        assert "0.40" in html  # sample 1 answer_relevancy
