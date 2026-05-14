"""Tests for py.typed marker and module exports."""

from __future__ import annotations

import os

import pytest


class TestPyTypedMarker:
    """Test that py.typed marker exists for PEP 561 compliance."""

    def test_py_typed_exists(self):
        """py.typed marker file should exist."""
        import llm_eval
        pkg_dir = os.path.dirname(llm_eval.__file__)
        py_typed = os.path.join(pkg_dir, "py.typed")
        assert os.path.exists(py_typed), f"py.typed not found at {py_typed}"

    def test_py_typed_is_file(self):
        """py.typed should be a regular file."""
        import llm_eval
        pkg_dir = os.path.dirname(llm_eval.__file__)
        py_typed = os.path.join(pkg_dir, "py.typed")
        assert os.path.isfile(py_typed)


class TestModuleExports:
    """Test that __all__ exports are properly defined."""

    def test_init_exports(self):
        """llm_eval.__init__ should export core classes."""
        import llm_eval
        expected = [
            "Sample", "MetricResult", "EvalResult",
            "EvalConfig", "JudgeConfig", "evaluate",
            "evaluate_file", "EvalOutput",
        ]
        for name in expected:
            assert name in llm_eval.__all__, f"{name} not in llm_eval.__all__"

    def test_dataset_exports(self):
        """llm_eval.dataset should export key functions."""
        from llm_eval import dataset
        assert "load_jsonl" in dataset.__all__
        assert "load_csv" in dataset.__all__
        assert "load_dataset" in dataset.__all__
        assert "stream_jsonl" in dataset.__all__
        assert "count_samples" in dataset.__all__

    def test_metrics_exports(self):
        """llm_eval.metrics should export key classes."""
        from llm_eval import metrics
        assert "Metric" in metrics.__all__
        assert "MetricResult" in metrics.__all__
        assert "MetricRegistry" in metrics.__all__
        assert "get_default_registry" in metrics.__all__

    def test_judge_exports(self):
        """llm_eval.judge should export Judge."""
        from llm_eval import judge
        assert "Judge" in judge.__all__

    def test_version_exists(self):
        """llm_eval should have __version__."""
        from llm_eval import __version__
        assert isinstance(__version__, str)
        assert "." in __version__


class TestDatasetExports:
    """Test that new dataset functions are importable."""

    def test_stream_jsonl_importable(self):
        """stream_jsonl should be importable from dataset module."""
        from llm_eval.dataset import stream_jsonl
        assert callable(stream_jsonl)

    def test_count_samples_importable(self):
        """count_samples should be importable from dataset module."""
        from llm_eval.dataset import count_samples
        assert callable(count_samples)
