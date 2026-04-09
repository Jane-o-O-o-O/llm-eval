"""Tests for the dataset loader."""

import json

import pytest

from llm_eval.dataset import load_jsonl
from llm_eval.models import Sample


class TestLoadJsonl:
    """Tests for the JSONL dataset loader."""

    def test_load_basic_jsonl(self, tmp_path) -> None:
        data = [
            {"query": "q1", "context": ["c1"], "answer": "a1"},
            {"query": "q2", "context": ["c2"], "answer": "a2"},
        ]
        filepath = tmp_path / "test.jsonl"
        filepath.write_text("\n".join(json.dumps(d) for d in data) + "\n")

        samples = load_jsonl(str(filepath))
        assert len(samples) == 2
        assert isinstance(samples[0], Sample)
        assert samples[0].query == "q1"
        assert samples[1].query == "q2"

    def test_load_jsonl_with_reference(self, tmp_path) -> None:
        data = [
            {"query": "q1", "context": ["c1"], "answer": "a1", "reference": "ref1"},
        ]
        filepath = tmp_path / "test.jsonl"
        filepath.write_text(json.dumps(data[0]) + "\n")

        samples = load_jsonl(str(filepath))
        assert samples[0].reference == "ref1"

    def test_load_jsonl_with_metadata(self, tmp_path) -> None:
        data = [
            {
                "query": "q1",
                "context": ["c1"],
                "answer": "a1",
                "metadata": {"version": "v2"},
            },
        ]
        filepath = tmp_path / "test.jsonl"
        filepath.write_text(json.dumps(data[0]) + "\n")

        samples = load_jsonl(str(filepath))
        assert samples[0].metadata == {"version": "v2"}

    def test_load_jsonl_empty_file(self, tmp_path) -> None:
        filepath = tmp_path / "empty.jsonl"
        filepath.write_text("")

        samples = load_jsonl(str(filepath))
        assert samples == []

    def test_load_jsonl_missing_required_field(self, tmp_path) -> None:
        filepath = tmp_path / "bad.jsonl"
        filepath.write_text(json.dumps({"query": "q1"}) + "\n")

        with pytest.raises(ValueError, match="Missing required field"):
            load_jsonl(str(filepath))

    def test_load_jsonl_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_jsonl("/nonexistent/file.jsonl")

    def test_load_jsonl_malformed_json(self, tmp_path) -> None:
        filepath = tmp_path / "malformed.jsonl"
        filepath.write_text("not valid json\n")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_jsonl(str(filepath))

    def test_load_jsonl_blank_lines_ignored(self, tmp_path) -> None:
        data = {"query": "q1", "context": ["c1"], "answer": "a1"}
        filepath = tmp_path / "test.jsonl"
        filepath.write_text(json.dumps(data) + "\n\n\n")

        samples = load_jsonl(str(filepath))
        assert len(samples) == 1

    def test_load_jsonl_multiple_context_chunks(self, tmp_path) -> None:
        data = {
            "query": "q1",
            "context": ["chunk1", "chunk2", "chunk3"],
            "answer": "a1",
        }
        filepath = tmp_path / "test.jsonl"
        filepath.write_text(json.dumps(data) + "\n")

        samples = load_jsonl(str(filepath))
        assert len(samples[0].context) == 3

# [2026-04-09] Tests for test_dataset
class TestTestDataset:
    """Test suite for test_dataset — custom metric registration."""

    def setup_method(self):
        """Setup test fixtures."""
        self.fixture = {}
        self.config = {"enabled": True, "debug": False}

    def test_basic_custom_metric_registration(self):
        """Test basic custom metric registration functionality."""
        result = process(self.fixture, config=self.config)
        assert result is not None
        assert result.get("status") == "success"

    def test_custom_metric_registration_with_empty_input(self):
        """Test custom metric registration with empty input."""
        result = process({}, config=self.config)
        assert result is not None

    def test_custom_metric_registration_error_handling(self):
        """Test custom metric registration error handling."""
        with pytest.raises(ValueError):
            process(None, config=self.config)

    def test_custom_metric_registration_caching(self):
        """Test custom metric registration caching behavior."""
        result1 = process(self.fixture, config=self.config)
        result2 = process(self.fixture, config=self.config)
        assert result1 == result2
