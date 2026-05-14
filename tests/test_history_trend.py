"""Tests for history trend command."""

from __future__ import annotations

import json
import os
import tempfile

from click.testing import CliRunner

from llm_eval.cli import main


class TestHistoryTrend:
    """Test history trend display."""

    def _make_history_run(self, hdir: str, name: str, score: float) -> None:
        data = {
            "timestamp": "2026-01-01T00:00:00Z",
            "tag": "test",
            "summary": {
                "overall_score": score,
                "total_samples": 10,
                "pass_count": 8,
                "fail_count": 2,
            },
            "results": [],
        }
        with open(os.path.join(hdir, name), "w") as f:
            json.dump(data, f)

    def test_trend_with_runs(self):
        """Trend should display score history."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            self._make_history_run(tmpdir, "20260101_000000.json", 0.8)
            self._make_history_run(tmpdir, "20260102_000000.json", 0.85)
            self._make_history_run(tmpdir, "20260103_000000.json", 0.75)
            result = runner.invoke(
                main, ["history", "trend", "--history-dir", tmpdir]
            )
            assert result.exit_code == 0
            # Should show scores
            assert "0.8" in result.output or "0.80" in result.output

    def test_trend_empty(self):
        """Trend with no history should show message."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = runner.invoke(
                main, ["history", "trend", "--history-dir", tmpdir]
            )
            assert result.exit_code == 0
            assert "No" in result.output or "no" in result.output or "📭" in result.output

    def test_trend_help(self):
        """trend --help should work."""
        runner = CliRunner()
        result = runner.invoke(main, ["history", "trend", "--help"])
        assert result.exit_code == 0
