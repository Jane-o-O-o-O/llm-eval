"""Tests for the dataset CLI subcommand."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from llm_eval.cli import main


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def jsonl_dataset(tmp_path):
    path = tmp_path / "samples.jsonl"
    path.write_text(
        '{"query": "What is Python?", "context": ["Python is a programming language."], "answer": "A programming language.", "reference": "Python is a high-level programming language."}\n'
        '{"query": "What is Rust?", "context": ["Rust is a systems language."], "answer": "A systems programming language."}\n'
        '{"query": "Empty context test", "context": [], "answer": "Some answer."}\n'
    )
    return str(path)


@pytest.fixture
def csv_dataset(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text(
        "query,context,answer,reference\n"
        "What is Python?,Python is a language.,A language.,Python is a high-level language.\n"
        "What is Rust?,Rust is fast.,A fast language.,\n"
    )
    return str(path)


class TestDatasetInfo:
    """Tests for `llm-eval dataset info`."""

    def test_info_jsonl(self, runner, jsonl_dataset):
        result = runner.invoke(main, ["dataset", "info", jsonl_dataset])
        assert result.exit_code == 0
        assert "3" in result.output  # 3 samples
        assert "Dataset" in result.output

    def test_info_csv(self, runner, csv_dataset):
        result = runner.invoke(main, ["dataset", "info", csv_dataset])
        assert result.exit_code == 0
        assert "2" in result.output  # 2 samples

    def test_info_shows_statistics(self, runner, jsonl_dataset):
        result = runner.invoke(main, ["dataset", "info", jsonl_dataset])
        assert "With reference" in result.output
        assert "With context" in result.output
        assert "Avg context chunks" in result.output

    def test_info_nonexistent_file(self, runner):
        result = runner.invoke(main, ["dataset", "info", "/nonexistent.jsonl"])
        assert result.exit_code != 0


class TestDatasetValidate:
    """Tests for `llm-eval dataset validate`."""

    def test_valid_dataset(self, runner, jsonl_dataset):
        result = runner.invoke(main, ["dataset", "validate", jsonl_dataset])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()

    def test_empty_query_fails(self, runner, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"query": "", "context": ["C"], "answer": "A"}\n')
        result = runner.invoke(main, ["dataset", "validate", str(path)])
        assert result.exit_code != 0
        assert "Empty query" in result.output

    def test_empty_answer_fails(self, runner, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"query": "Q", "context": ["C"], "answer": ""}\n')
        result = runner.invoke(main, ["dataset", "validate", str(path)])
        assert result.exit_code != 0
        assert "Empty answer" in result.output

    def test_no_context_warns(self, runner, tmp_path):
        path = tmp_path / "noctx.jsonl"
        path.write_text('{"query": "Q", "context": [], "answer": "A"}\n')
        result = runner.invoke(main, ["dataset", "validate", str(path)])
        assert result.exit_code == 0
        assert "No context" in result.output

    def test_query_equals_answer_warns(self, runner, tmp_path):
        path = tmp_path / "same.jsonl"
        path.write_text('{"query": "same", "context": ["C"], "answer": "same"}\n')
        result = runner.invoke(main, ["dataset", "validate", str(path)])
        assert result.exit_code == 0
        assert "identical" in result.output.lower()


class TestDatasetSample:
    """Tests for `llm-eval dataset sample`."""

    def test_show_samples(self, runner, jsonl_dataset):
        result = runner.invoke(main, ["dataset", "sample", jsonl_dataset])
        assert result.exit_code == 0
        assert "Query:" in result.output
        assert "Answer:" in result.output

    def test_show_n_samples(self, runner, jsonl_dataset):
        result = runner.invoke(main, ["dataset", "sample", "-n", "1", jsonl_dataset])
        assert result.exit_code == 0
        # Should show exactly 1 sample
        assert result.output.count("Sample #") == 1

    def test_reproducible_with_seed(self, runner, jsonl_dataset):
        r1 = runner.invoke(main, ["dataset", "sample", "-n", "2", "--seed", "42", jsonl_dataset])
        r2 = runner.invoke(main, ["dataset", "sample", "-n", "2", "--seed", "42", jsonl_dataset])
        assert r1.output == r2.output

    def test_csv_dataset(self, runner, csv_dataset):
        result = runner.invoke(main, ["dataset", "sample", csv_dataset])
        assert result.exit_code == 0
        assert "Query:" in result.output


class TestDatasetGroupHelp:
    """Tests for the dataset group help text."""

    def test_dataset_help(self, runner):
        result = runner.invoke(main, ["dataset", "--help"])
        assert result.exit_code == 0
        assert "info" in result.output
        assert "validate" in result.output
        assert "sample" in result.output
