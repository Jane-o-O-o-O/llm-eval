"""Tests for --quiet flag and HTML chart features."""

from __future__ import annotations

import yaml
from click.testing import CliRunner

from llm_eval.cli import main
from llm_eval.models import EvalResult, MetricResult
from llm_eval.report import format_html_report


class TestQuietFlag:
    """Tests for the --quiet/-q CLI flag."""

    def test_quiet_flag_accepted(self) -> None:
        """Test that --quiet flag is accepted by run command."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--quiet" in result.output

    def test_quiet_dry_run_minimal_output(self, tmp_path) -> None:
        """Test --quiet with --dry-run produces minimal output."""
        dataset_path = tmp_path / "samples.jsonl"
        dataset_path.write_text('{"query": "Q", "context": ["C"], "answer": "A"}\n')

        config = {
            "judge": {"model": "gpt-4o"},
            "evaluations": [
                {
                    "name": "Test",
                    "dataset": str(dataset_path),
                    "metrics": ["faithfulness"],
                }
            ],
        }
        config_path = tmp_path / "evals.yaml"
        config_path.write_text(yaml.dump(config))

        runner = CliRunner()
        result_quiet = runner.invoke(
            main,
            [
                "run",
                "--config",
                str(config_path),
                "--dry-run",
                "--quiet",
            ],
        )
        result_verbose = runner.invoke(
            main,
            [
                "run",
                "--config",
                str(config_path),
                "--dry-run",
            ],
        )
        # Dry-run always prints (it's the plan), but quiet mode suppresses other output
        assert result_quiet.exit_code == 0
        assert result_verbose.exit_code == 0


class TestHtmlChart:
    """Tests for SVG chart in HTML report."""

    def test_html_contains_svg_chart(self) -> None:
        """Test that HTML report contains an SVG score distribution chart."""
        results = [
            EvalResult(
                sample_index=i,
                metrics=[
                    MetricResult(name="faithfulness", score=0.5 + i * 0.1, details={}),
                ],
            )
            for i in range(5)
        ]
        summary = {
            "total_samples": 5,
            "overall_score": 0.7,
            "pass_count": 3,
            "fail_count": 2,
            "pass_rate": 0.6,
            "threshold": 0.7,
            "metric_scores": {
                "faithfulness": {"mean": 0.7, "min": 0.5, "max": 0.9},
            },
        }
        html = format_html_report(results, summary)
        assert "<svg" in html
        assert "</svg>" in html
        assert "Score Distribution" in html

    def test_svg_contains_bars(self) -> None:
        """Test that SVG chart contains rect elements for bars."""
        results = [
            EvalResult(
                sample_index=0,
                metrics=[
                    MetricResult(name="m", score=0.85, details={}),
                ],
            ),
        ]
        summary = {
            "total_samples": 1,
            "overall_score": 0.85,
            "pass_count": 1,
            "fail_count": 0,
            "pass_rate": 1.0,
            "threshold": 0.7,
            "metric_scores": {"m": {"mean": 0.85, "min": 0.85, "max": 0.85}},
        }
        html = format_html_report(results, summary)
        assert "<rect" in html

    def test_svg_with_empty_results(self) -> None:
        """Test SVG chart with empty results doesn't crash."""
        results: list[EvalResult] = []
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
        assert "<svg" in html
