"""Tests for evaluation run history (llm_eval.history)."""

from __future__ import annotations

import json
import os

import pytest

from llm_eval.history import list_runs, load_run, save_run
from llm_eval.models import EvalResult, MetricResult


@pytest.fixture
def mock_results():
    return [
        EvalResult(
            sample_index=0,
            metrics=[MetricResult(name="faithfulness", score=0.9)],
        ),
    ]


@pytest.fixture
def summary():
    return {
        "total_samples": 1,
        "overall_score": 0.9,
        "pass_count": 1,
        "fail_count": 0,
        "pass_rate": 1.0,
        "threshold": 0.7,
        "metric_scores": {"faithfulness": {"mean": 0.9, "min": 0.9, "max": 0.9}},
    }


class TestSaveRun:
    """Tests for save_run()."""

    def test_save_creates_file(self, tmp_path, mock_results, summary):
        path = save_run(mock_results, summary, history_dir=str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".json")

    def test_save_with_tag(self, tmp_path, mock_results, summary):
        path = save_run(mock_results, summary, tag="baseline", history_dir=str(tmp_path))
        assert "baseline" in path

    def test_save_without_tag(self, tmp_path, mock_results, summary):
        path = save_run(mock_results, summary, history_dir=str(tmp_path))
        assert "baseline" not in path

    def test_save_contains_summary(self, tmp_path, mock_results, summary):
        path = save_run(mock_results, summary, history_dir=str(tmp_path))
        with open(path) as f:
            data = json.load(f)
        assert data["summary"]["overall_score"] == 0.9
        assert data["summary"]["total_samples"] == 1

    def test_save_contains_results(self, tmp_path, mock_results, summary):
        path = save_run(mock_results, summary, history_dir=str(tmp_path))
        with open(path) as f:
            data = json.load(f)
        assert len(data["results"]) == 1
        assert data["results"][0]["sample_index"] == 0

    def test_save_contains_timestamp(self, tmp_path, mock_results, summary):
        path = save_run(mock_results, summary, history_dir=str(tmp_path))
        with open(path) as f:
            data = json.load(f)
        assert "timestamp" in data

    def test_save_with_config_path(self, tmp_path, mock_results, summary):
        path = save_run(mock_results, summary, config_path="evals.yaml", history_dir=str(tmp_path))
        with open(path) as f:
            data = json.load(f)
        assert data["config_path"] == "evals.yaml"

    def test_save_with_tag_in_data(self, tmp_path, mock_results, summary):
        path = save_run(mock_results, summary, tag="experiment-1", history_dir=str(tmp_path))
        with open(path) as f:
            data = json.load(f)
        assert data["tag"] == "experiment-1"


class TestListRuns:
    """Tests for list_runs()."""

    def test_empty_history(self, tmp_path):
        runs = list_runs(history_dir=str(tmp_path))
        assert runs == []

    def test_list_after_save(self, tmp_path, mock_results, summary):
        save_run(mock_results, summary, history_dir=str(tmp_path))
        runs = list_runs(history_dir=str(tmp_path))
        assert len(runs) == 1
        assert "overall_score" in runs[0]
        assert runs[0]["overall_score"] == 0.9

    def test_list_multiple_runs(self, tmp_path, mock_results, summary):
        save_run(mock_results, summary, tag="run-a", history_dir=str(tmp_path))
        save_run(mock_results, summary, tag="run-b", history_dir=str(tmp_path))
        runs = list_runs(history_dir=str(tmp_path))
        assert len(runs) == 2

    def test_filter_by_tag(self, tmp_path, mock_results, summary):
        save_run(mock_results, summary, history_dir=str(tmp_path))
        save_run(mock_results, summary, tag="baseline", history_dir=str(tmp_path))
        save_run(mock_results, summary, tag="experiment", history_dir=str(tmp_path))

        baseline_runs = list_runs(tag="baseline", history_dir=str(tmp_path))
        assert len(baseline_runs) == 1
        assert baseline_runs[0]["tag"] == "baseline"

    def test_limit(self, tmp_path, mock_results, summary):
        for i in range(5):
            save_run(mock_results, summary, tag=f"run-{i}", history_dir=str(tmp_path))
        runs = list_runs(limit=3, history_dir=str(tmp_path))
        assert len(runs) == 3

    def test_runs_sorted_newest_first(self, tmp_path, mock_results, summary):
        save_run(mock_results, summary, tag="first", history_dir=str(tmp_path))
        save_run(mock_results, summary, tag="second", history_dir=str(tmp_path))
        runs = list_runs(history_dir=str(tmp_path))
        assert len(runs) == 2
        # Both runs should be present
        tags = [r["tag"] for r in runs]
        assert "first" in tags
        assert "second" in tags

    def test_run_has_path_and_file(self, tmp_path, mock_results, summary):
        save_run(mock_results, summary, history_dir=str(tmp_path))
        runs = list_runs(history_dir=str(tmp_path))
        assert "path" in runs[0]
        assert "file" in runs[0]
        assert runs[0]["file"].endswith(".json")


class TestLoadRun:
    """Tests for load_run()."""

    def test_load_returns_full_data(self, tmp_path, mock_results, summary):
        path = save_run(mock_results, summary, tag="test", history_dir=str(tmp_path))
        data = load_run(path)
        assert "summary" in data
        assert "results" in data
        assert "timestamp" in data
        assert data["tag"] == "test"

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_run("/nonexistent/run.json")

    def test_roundtrip(self, tmp_path, mock_results, summary):
        path = save_run(mock_results, summary, history_dir=str(tmp_path))
        data = load_run(path)
        assert data["summary"]["overall_score"] == summary["overall_score"]
        assert len(data["results"]) == len(mock_results)
