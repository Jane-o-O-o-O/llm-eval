"""Tests for CSV dataset loader."""

from __future__ import annotations

import pytest

from llm_eval.dataset import load_csv, load_dataset
from llm_eval.models import Sample


class TestLoadCsv:
    """Tests for the CSV dataset loader."""

    def test_load_basic_csv(self, tmp_path) -> None:
        filepath = tmp_path / "test.csv"
        filepath.write_text(
            "query,context,answer\n"
            "What is AI?,Artificial intelligence is,AI is a field of CS\n"
            "What is ML?,Machine learning is,ML is a subset of AI\n"
        )
        samples = load_csv(str(filepath))
        assert len(samples) == 2
        assert isinstance(samples[0], Sample)
        assert samples[0].query == "What is AI?"
        assert samples[1].query == "What is ML?"

    def test_load_csv_pipe_separated_context(self, tmp_path) -> None:
        filepath = tmp_path / "test.csv"
        filepath.write_text(
            "query,context,answer\n"
            "What is AI?,AI is a field | It studies intelligence,AI answer\n"
        )
        samples = load_csv(str(filepath))
        assert samples[0].context == ["AI is a field", "It studies intelligence"]

    def test_load_csv_json_array_context(self, tmp_path) -> None:
        filepath = tmp_path / "test.csv"
        filepath.write_text(
            "query,context,answer\n"
            'What is AI?,"[""chunk1"", ""chunk2""]",AI answer\n'
        )
        samples = load_csv(str(filepath))
        assert samples[0].context == ["chunk1", "chunk2"]

    def test_load_csv_with_reference(self, tmp_path) -> None:
        filepath = tmp_path / "test.csv"
        filepath.write_text(
            "query,context,answer,reference\n"
            "What is AI?,context,AI answer,reference answer\n"
        )
        samples = load_csv(str(filepath))
        assert samples[0].reference == "reference answer"

    def test_load_csv_with_metadata(self, tmp_path) -> None:
        filepath = tmp_path / "test.csv"
        filepath.write_text(
            "query,context,answer,metadata\n"
            'What is AI?,context,AI answer,"{""expected_format"": ""json""}"\n'
        )
        samples = load_csv(str(filepath))
        assert samples[0].metadata == {"expected_format": "json"}

    def test_load_csv_empty_context(self, tmp_path) -> None:
        filepath = tmp_path / "test.csv"
        filepath.write_text("query,context,answer\nWhat is AI?,,AI answer\n")
        samples = load_csv(str(filepath))
        assert samples[0].context == []

    def test_load_csv_missing_required_column(self, tmp_path) -> None:
        filepath = tmp_path / "test.csv"
        filepath.write_text("query,answer\nWhat is AI?,AI answer\n")
        with pytest.raises(ValueError, match="missing required column"):
            load_csv(str(filepath))

    def test_load_csv_no_data_rows(self, tmp_path) -> None:
        filepath = tmp_path / "test.csv"
        filepath.write_text("query,context,answer\n")
        with pytest.raises(ValueError, match="no data rows"):
            load_csv(str(filepath))

    def test_load_csv_empty_reference_becomes_none(self, tmp_path) -> None:
        filepath = tmp_path / "test.csv"
        filepath.write_text("query,context,answer,reference\nWhat is AI?,ctx,AI ans,\n")
        samples = load_csv(str(filepath))
        assert samples[0].reference is None

    def test_load_csv_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_csv("/nonexistent/file.csv")

    def test_load_csv_invalid_metadata_json(self, tmp_path) -> None:
        filepath = tmp_path / "test.csv"
        filepath.write_text(
            "query,context,answer,metadata\n"
            "What is AI?,ctx,AI ans,not-json\n"
        )
        samples = load_csv(str(filepath))
        # Falls back to raw_metadata
        assert "raw_metadata" in samples[0].metadata


class TestLoadDataset:
    """Tests for the auto-detect dataset loader."""

    def test_auto_detect_jsonl(self, tmp_path) -> None:
        filepath = tmp_path / "test.jsonl"
        filepath.write_text('{"query": "q", "context": ["c"], "answer": "a"}\n')
        samples = load_dataset(str(filepath))
        assert len(samples) == 1

    def test_auto_detect_csv(self, tmp_path) -> None:
        filepath = tmp_path / "test.csv"
        filepath.write_text("query,context,answer\nq,c,a\n")
        samples = load_dataset(str(filepath))
        assert len(samples) == 1

    def test_auto_detect_ndjson(self, tmp_path) -> None:
        filepath = tmp_path / "test.ndjson"
        filepath.write_text('{"query": "q", "context": ["c"], "answer": "a"}\n')
        samples = load_dataset(str(filepath))
        assert len(samples) == 1

    def test_auto_detect_unknown_ext_defaults_to_jsonl(self, tmp_path) -> None:
        filepath = tmp_path / "test.txt"
        filepath.write_text('{"query": "q", "context": ["c"], "answer": "a"}\n')
        samples = load_dataset(str(filepath))
        assert len(samples) == 1

    def test_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            load_dataset("/nonexistent/file.jsonl")
