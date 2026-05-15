"""Tests for history diff command."""

from __future__ import annotations

import json

from click.testing import CliRunner

from llm_eval.cli import main


def _create_run(history_dir, name, score):
    """Create a minimal history run file."""
    path = history_dir / f"{name}.json"
    path.write_text(json.dumps({
        "timestamp": "2026-01-01T00:00:00+00:00",
        "tag": name.split("_", 1)[-1] if "_" in name else None,
        "summary": {
            "overall_score": score,
            "total_samples": 2,
            "pass_count": 2,
            "fail_count": 0,
            "pass_rate": 1.0,
            "threshold": 0.7,
            "metric_scores": {
                "faithfulness": {"mean": score, "min": score, "max": score},
            },
            "median": score,
            "p25": score,
            "p75": score,
            "std_dev": 0.0,
            "min_score": score,
            "max_score": score,
        },
        "results": [],
    }))
    return path


class TestHistoryDiff:
    """Test history diff command."""

    def test_diff_two_runs(self, tmp_path):
        hdir = tmp_path / "history"
        hdir.mkdir()
        p1 = _create_run(hdir, "20260101_000000_baseline", 0.90)
        p2 = _create_run(hdir, "20260102_000000_current", 0.85)

        runner = CliRunner()
        result = runner.invoke(main, [
            "history", "diff", str(p1), str(p2),
        ])
        assert result.exit_code == 0
        assert "faithfulness" in result.output

    def test_diff_missing_run(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, [
            "history", "diff",
            "/nonexistent/run1.json",
            "/nonexistent/run2.json",
        ])
        assert result.exit_code != 0

    def test_diff_output_json(self, tmp_path):
        hdir = tmp_path / "history"
        hdir.mkdir()
        p1 = _create_run(hdir, "20260101_000000_baseline", 0.90)
        p2 = _create_run(hdir, "20260102_000000_current", 0.85)

        runner = CliRunner()
        result = runner.invoke(main, [
            "history", "diff", str(p1), str(p2),
            "--output", "json",
        ])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "comparisons" in data
