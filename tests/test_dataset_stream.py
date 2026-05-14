"""Tests for streaming dataset loader and count_samples."""

from __future__ import annotations

import json
import os

import pytest

from llm_eval.dataset import count_samples, load_jsonl, stream_jsonl
from llm_eval.models import Sample


@pytest.fixture
def jsonl_file(tmp_path):
    """Create a temporary JSONL file with test samples."""
    filepath = tmp_path / "test.jsonl"
    samples = [
        {"query": f"Q{i}", "context": [f"C{i}"], "answer": f"A{i}"}
        for i in range(10)
    ]
    with open(filepath, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    return str(filepath)


@pytest.fixture
def csv_file(tmp_path):
    """Create a temporary CSV file with test samples."""
    filepath = tmp_path / "test.csv"
    with open(filepath, "w") as f:
        f.write("query,context,answer,reference\n")
        for i in range(5):
            f.write(f'Q{i},"C{i}",A{i},R{i}\n')
    return str(filepath)


class TestStreamJsonl:
    """Test the stream_jsonl function."""

    def test_stream_returns_generator(self, jsonl_file):
        """stream_jsonl should return a generator."""
        gen = stream_jsonl(jsonl_file)
        # Should be iterable
        assert hasattr(gen, "__iter__")
        assert hasattr(gen, "__next__")

    def test_stream_yields_all_samples(self, jsonl_file):
        """stream_jsonl should yield all samples."""
        samples = list(stream_jsonl(jsonl_file))
        assert len(samples) == 10

    def test_stream_yields_sample_objects(self, jsonl_file):
        """stream_jsonl should yield Sample objects."""
        for sample in stream_jsonl(jsonl_file):
            assert isinstance(sample, Sample)

    def test_stream_lazy_evaluation(self, jsonl_file):
        """stream_jsonl should not load all samples at once."""
        gen = stream_jsonl(jsonl_file)
        # Get first sample
        first = next(gen)
        assert first.query == "Q0"
        assert first.answer == "A0"
        # Generator should still have more items
        second = next(gen)
        assert second.query == "Q1"

    def test_stream_skips_blank_lines(self, tmp_path):
        """stream_jsonl should skip empty lines."""
        filepath = tmp_path / "blank.jsonl"
        with open(filepath, "w") as f:
            f.write(json.dumps({"query": "Q1", "context": [], "answer": "A1"}) + "\n")
            f.write("\n")
            f.write("   \n")
            f.write(json.dumps({"query": "Q2", "context": [], "answer": "A2"}) + "\n")

        samples = list(stream_jsonl(str(filepath)))
        assert len(samples) == 2

    def test_stream_invalid_json_raises(self, tmp_path):
        """stream_jsonl should raise ValueError for invalid JSON."""
        filepath = tmp_path / "bad.jsonl"
        with open(filepath, "w") as f:
            f.write("not valid json\n")

        with pytest.raises(ValueError, match="Invalid JSON on line 1"):
            list(stream_jsonl(str(filepath)))

    def test_stream_missing_fields_raises(self, tmp_path):
        """stream_jsonl should raise ValueError for missing required fields."""
        filepath = tmp_path / "incomplete.jsonl"
        with open(filepath, "w") as f:
            f.write(json.dumps({"query": "Q1"}) + "\n")

        with pytest.raises(ValueError, match="Missing required field"):
            list(stream_jsonl(str(filepath)))

    def test_stream_file_not_found(self):
        """stream_jsonl should raise FileNotFoundError for missing files."""
        with pytest.raises(FileNotFoundError):
            list(stream_jsonl("/nonexistent/file.jsonl"))

    def test_stream_with_reference(self, tmp_path):
        """stream_jsonl should handle reference field."""
        filepath = tmp_path / "ref.jsonl"
        with open(filepath, "w") as f:
            f.write(json.dumps({
                "query": "Q1",
                "context": ["C1"],
                "answer": "A1",
                "reference": "R1",
            }) + "\n")

        samples = list(stream_jsonl(str(filepath)))
        assert samples[0].reference == "R1"


class TestCountSamples:
    """Test the count_samples function."""

    def test_count_jsonl(self, jsonl_file):
        """count_samples should count JSONL samples."""
        assert count_samples(jsonl_file) == 10

    def test_count_csv(self, csv_file):
        """count_samples should count CSV samples."""
        assert count_samples(csv_file) == 5

    def test_count_empty_jsonl(self, tmp_path):
        """count_samples should return 0 for empty file."""
        filepath = tmp_path / "empty.jsonl"
        filepath.touch()
        assert count_samples(str(filepath)) == 0

    def test_count_skips_blank_lines(self, tmp_path):
        """count_samples should skip blank lines in JSONL."""
        filepath = tmp_path / "blank.jsonl"
        with open(filepath, "w") as f:
            f.write('{"query": "Q1", "context": [], "answer": "A1"}\n')
            f.write("\n")
            f.write('{"query": "Q2", "context": [], "answer": "A2"}\n')
        assert count_samples(str(filepath)) == 2

    def test_count_file_not_found(self):
        """count_samples should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            count_samples("/nonexistent/file.jsonl")

    def test_count_ndjson_extension(self, tmp_path):
        """count_samples should work with .ndjson extension."""
        filepath = tmp_path / "test.ndjson"
        with open(filepath, "w") as f:
            f.write('{"query": "Q1", "context": [], "answer": "A1"}\n')
        assert count_samples(str(filepath)) == 1
