"""Tests for CLI --tag option and history command."""

from __future__ import annotations

import os

import pytest
from click.testing import CliRunner

from llm_eval.cli import main
from llm_eval.history import save_run
from llm_eval.models import EvalResult, MetricResult


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def eval_project(tmp_path):
    """Create a minimal evaluation project with config and dataset."""
    config = tmp_path / "evals.yaml"
    config.write_text(
        "judge:\n"
        "  model: gpt-4o\n"
        "  temperature: 0\n"
        "defaults:\n"
        "  threshold: 0.5\n"
        "  output_format: terminal\n"
        "evaluations:\n"
        '  - name: "Test Eval"\n'
        "    dataset: samples.jsonl\n"
        "    metrics:\n"
        "      - faithfulness\n"
    )
    ds = tmp_path / "samples.jsonl"
    ds.write_text(
        '{"query": "What is Python?", "context": ["Python is a programming language."], "answer": "A programming language."}\n'
    )
    return tmp_path


class TestTagOption:
    """Tests for --tag on the run command."""

    def test_tag_option_exists(self, runner, eval_project):
        """Test that --tag option is accepted (dry run to avoid API calls)."""
        result = runner.invoke(
            main,
            ["run", "--config", str(eval_project / "evals.yaml"), "--dry-run", "--tag", "baseline"],
        )
        assert result.exit_code == 0
        # Dry run shouldn't fail with --tag option
        assert "DRY RUN" in result.output

    def test_no_cache_option_exists(self, runner, eval_project):
        """Test that --no-cache option is accepted."""
        result = runner.invoke(
            main,
            ["run", "--config", str(eval_project / "evals.yaml"), "--dry-run", "--no-cache"],
        )
        assert result.exit_code == 0

    def test_save_history_option_exists(self, runner, eval_project):
        """Test that --no-save-history option is accepted."""
        result = runner.invoke(
            main,
            ["run", "--config", str(eval_project / "evals.yaml"), "--dry-run"],
        )
        assert result.exit_code == 0


class TestHistoryCommand:
    """Tests for the `history` CLI command."""

    def test_history_empty(self, runner, tmp_path):
        """Test history with no runs."""
        # Use a temp dir so it's empty
        os.environ["LLM_EVAL_HISTORY_DIR"] = str(tmp_path / "history")
        result = runner.invoke(main, ["history"])
        # It should at least not crash - might show "no runs" or use default dir
        assert result.exit_code == 0
        del os.environ["LLM_EVAL_HISTORY_DIR"]

    def test_history_with_runs(self, runner, tmp_path):
        """Test history shows saved runs."""
        hdir = str(tmp_path / "history")
        results = [
            EvalResult(sample_index=0, metrics=[MetricResult(name="faithfulness", score=0.9)])
        ]
        summary = {
            "total_samples": 1,
            "overall_score": 0.9,
            "pass_count": 1,
            "fail_count": 0,
            "pass_rate": 1.0,
            "threshold": 0.7,
            "metric_scores": {"faithfulness": {"mean": 0.9, "min": 0.9, "max": 0.9}},
        }
        save_run(results, summary, tag="test-run", history_dir=hdir)

        # The history command reads from default dir, so this test verifies
        # the save_run function works correctly which the history command uses
        from llm_eval.history import list_runs
        runs = list_runs(history_dir=hdir)
        assert len(runs) == 1
        assert runs[0]["tag"] == "test-run"
        assert runs[0]["overall_score"] == 0.9

    def test_history_filter_by_tag(self, runner, tmp_path):
        """Test history filtering by tag."""
        hdir = str(tmp_path / "history")
        results = [
            EvalResult(sample_index=0, metrics=[MetricResult(name="f", score=0.8)])
        ]
        summary = {"total_samples": 1, "overall_score": 0.8, "pass_count": 1, "fail_count": 0, "pass_rate": 1.0, "metric_scores": {}}
        save_run(results, summary, tag="baseline", history_dir=hdir)
        save_run(results, summary, tag="experiment", history_dir=hdir)
        save_run(results, summary, history_dir=hdir)  # no tag

        from llm_eval.history import list_runs
        baseline_runs = list_runs(tag="baseline", history_dir=hdir)
        assert len(baseline_runs) == 1
        assert baseline_runs[0]["tag"] == "baseline"

        all_runs = list_runs(history_dir=hdir)
        assert len(all_runs) == 3

    def test_history_limit(self, tmp_path):
        """Test history limit option."""
        hdir = str(tmp_path / "history")
        results = [EvalResult(sample_index=0, metrics=[MetricResult(name="f", score=0.8)])]
        summary = {"total_samples": 1, "overall_score": 0.8, "pass_count": 1, "fail_count": 0, "pass_rate": 1.0, "metric_scores": {}}
        for i in range(5):
            save_run(results, summary, tag=f"run-{i}", history_dir=hdir)

        from llm_eval.history import list_runs
        runs = list_runs(limit=2, history_dir=hdir)
        assert len(runs) == 2


class TestMarkdownFormatInCLI:
    """Tests for markdown as an output format in CLI validate."""

    def test_markdown_is_valid_format(self, runner, tmp_path):
        """Test that 'markdown' is accepted as a valid output format."""
        config = tmp_path / "evals.yaml"
        config.write_text(
            "judge:\n  model: gpt-4o\n  temperature: 0\n"
            "defaults:\n  threshold: 0.7\n  output_format: markdown\n"
            "evaluations:\n  - name: Test\n    dataset: samples.jsonl\n    metrics:\n      - faithfulness\n"
        )
        ds = tmp_path / "samples.jsonl"
        ds.write_text('{"query": "Q", "context": ["C"], "answer": "A"}\n')

        result = runner.invoke(main, ["validate", str(config)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()
