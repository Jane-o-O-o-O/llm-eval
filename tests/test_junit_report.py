"""Tests for JUnit XML report format."""

from __future__ import annotations

import xml.etree.ElementTree as ET

from llm_eval.models import EvalResult, MetricResult
from llm_eval.report import format_junit_report


def _make_result(index: int, scores: list[tuple[str, float]]) -> EvalResult:
    """Helper to create an EvalResult with named scores."""
    metrics = [MetricResult(name=n, score=s) for n, s in scores]
    return EvalResult(sample_index=index, metrics=metrics)


class TestJunitReport:
    """Test JUnit XML report generation."""

    def test_valid_xml(self):
        """Output must be valid XML."""
        results = [_make_result(0, [("faithfulness", 0.9)])]
        summary = {
            "total_samples": 1,
            "overall_score": 0.9,
            "pass_count": 1,
            "fail_count": 0,
            "threshold": 0.7,
            "metric_scores": {"faithfulness": {"mean": 0.9, "min": 0.9, "max": 0.9}},
        }
        xml_str = format_junit_report(results, summary)
        # Should not raise
        ET.fromstring(xml_str)

    def test_testsuites_attributes(self):
        """Root <testsuites> should have correct counts."""
        results = [
            _make_result(0, [("faithfulness", 0.9)]),
            _make_result(1, [("faithfulness", 0.3)]),
        ]
        summary = {
            "total_samples": 2,
            "overall_score": 0.6,
            "pass_count": 1,
            "fail_count": 1,
            "threshold": 0.7,
            "metric_scores": {"faithfulness": {"mean": 0.6, "min": 0.3, "max": 0.9}},
        }
        xml_str = format_junit_report(results, summary)
        root = ET.fromstring(xml_str)
        assert root.tag == "testsuites"
        assert root.get("tests") == "2"
        assert root.get("failures") == "1"
        assert root.get("name") == "llm-eval"

    def test_testsuite_per_metric(self):
        """Each metric should generate a <testsuite>."""
        results = [
            _make_result(0, [("faithfulness", 0.9), ("relevancy", 0.8)]),
        ]
        summary = {
            "total_samples": 1,
            "overall_score": 0.85,
            "pass_count": 1,
            "fail_count": 0,
            "threshold": 0.7,
            "metric_scores": {
                "faithfulness": {"mean": 0.9, "min": 0.9, "max": 0.9},
                "relevancy": {"mean": 0.8, "min": 0.8, "max": 0.8},
            },
        }
        xml_str = format_junit_report(results, summary)
        root = ET.fromstring(xml_str)
        suites = root.findall("testsuite")
        names = {s.get("name") for s in suites}
        assert "faithfulness" in names
        assert "relevancy" in names

    def test_failure_for_low_score(self):
        """Samples below threshold should generate <failure> elements."""
        results = [_make_result(0, [("faithfulness", 0.3)])]
        summary = {
            "total_samples": 1,
            "overall_score": 0.3,
            "pass_count": 0,
            "fail_count": 1,
            "threshold": 0.7,
            "metric_scores": {"faithfulness": {"mean": 0.3, "min": 0.3, "max": 0.3}},
        }
        xml_str = format_junit_report(results, summary)
        root = ET.fromstring(xml_str)
        failures = root.findall(".//failure")
        assert len(failures) >= 1
        assert "0.3" in (failures[0].get("message") or failures[0].text or "")

    def test_no_failure_for_high_score(self):
        """Samples above threshold should have no <failure> elements."""
        results = [_make_result(0, [("faithfulness", 0.95)])]
        summary = {
            "total_samples": 1,
            "overall_score": 0.95,
            "pass_count": 1,
            "fail_count": 0,
            "threshold": 0.7,
            "metric_scores": {"faithfulness": {"mean": 0.95, "min": 0.95, "max": 0.95}},
        }
        xml_str = format_junit_report(results, summary)
        root = ET.fromstring(xml_str)
        failures = root.findall(".//failure")
        assert len(failures) == 0

    def test_empty_results(self):
        """Empty results should still produce valid XML."""
        results: list[EvalResult] = []
        summary = {
            "total_samples": 0,
            "overall_score": 0.0,
            "pass_count": 0,
            "fail_count": 0,
            "threshold": 0.7,
            "metric_scores": {},
        }
        xml_str = format_junit_report(results, summary)
        root = ET.fromstring(xml_str)
        assert root.get("tests") == "0"

    def test_metadata_in_properties(self):
        """Metadata should be included as <properties>."""
        results = [_make_result(0, [("faithfulness", 0.9)])]
        summary = {
            "total_samples": 1,
            "overall_score": 0.9,
            "pass_count": 1,
            "fail_count": 0,
            "threshold": 0.7,
            "metric_scores": {"faithfulness": {"mean": 0.9, "min": 0.9, "max": 0.9}},
        }
        metadata = {"version": "0.8.0", "timestamp": "2026-01-01T00:00:00Z"}
        xml_str = format_junit_report(results, summary, metadata=metadata)
        root = ET.fromstring(xml_str)
        props = root.findall(".//property")
        prop_map = {p.get("name"): p.get("value") for p in props}
        assert prop_map.get("version") == "0.8.0"
        assert prop_map.get("timestamp") == "2026-01-01T00:00:00Z"

    def test_multiple_samples_multiple_metrics(self):
        """Multiple samples with multiple metrics should all appear."""
        results = [
            _make_result(0, [("faithfulness", 0.9), ("relevancy", 0.8)]),
            _make_result(1, [("faithfulness", 0.5), ("relevancy", 0.6)]),
            _make_result(2, [("faithfulness", 0.7), ("relevancy", 0.7)]),
        ]
        summary = {
            "total_samples": 3,
            "overall_score": 0.7,
            "pass_count": 2,
            "fail_count": 1,
            "threshold": 0.7,
            "metric_scores": {
                "faithfulness": {"mean": 0.7, "min": 0.5, "max": 0.9},
                "relevancy": {"mean": 0.7, "min": 0.6, "max": 0.8},
            },
        }
        xml_str = format_junit_report(results, summary)
        root = ET.fromstring(xml_str)
        testcases = root.findall(".//testcase")
        # 3 samples × 2 metrics = 6 testcases
        assert len(testcases) == 6
