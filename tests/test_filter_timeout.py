"""Tests for --filter and --timeout options on the run command."""

from __future__ import annotations

import json
import os

import pytest
import yaml
from click.testing import CliRunner

from llm_eval.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestFilterOption:
    """Test the --filter option for filtering dataset samples."""

    def test_filter_by_metadata_field(self, tmp_path, runner):
        """--filter metadata.category=tech filters samples in dry-run."""
        dataset_path = tmp_path / "samples.jsonl"
        config_path = tmp_path / "config.yaml"

        samples = [
            {"query": "What is Python?", "context": ["Python is a language."], "answer": "A lang.", "metadata": {"category": "tech"}},
            {"query": "What is cooking?", "context": ["Cooking is art."], "answer": "An art.", "metadata": {"category": "food"}},
            {"query": "What is Java?", "context": ["Java is a lang."], "answer": "Another lang.", "metadata": {"category": "tech"}},
        ]
        with open(dataset_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        config = {
            "judge": {"model": "gpt-4o"},
            "evaluations": [
                {
                    "name": "Filtered Test",
                    "dataset": str(dataset_path),
                    "metrics": ["faithfulness"],
                }
            ],
        }
        config_path.write_text(yaml.dump(config))

        result = runner.invoke(main, [
            "run", "-c", str(config_path), "--dry-run",
            "--filter", "metadata.category=tech",
        ])
        assert result.exit_code == 0
        assert "2" in result.output  # 2 tech samples

    def test_filter_invalid_format(self, tmp_path, runner):
        """--filter with invalid format shows error."""
        dataset_path = tmp_path / "samples.jsonl"
        config_path = tmp_path / "config.yaml"

        with open(dataset_path, "w") as f:
            f.write(json.dumps({"query": "q", "context": [], "answer": "a"}) + "\n")

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
        config_path.write_text(yaml.dump(config))

        result = runner.invoke(main, [
            "run", "-c", str(config_path), "--dry-run",
            "--filter", "invalid_format_no_equals",
        ])
        assert result.exit_code != 0

    def test_filter_no_match(self, tmp_path, runner):
        """--filter with no matching samples shows 0."""
        dataset_path = tmp_path / "samples.jsonl"
        config_path = tmp_path / "config.yaml"

        samples = [
            {"query": "q", "context": ["c"], "answer": "a", "metadata": {"category": "food"}},
        ]
        with open(dataset_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

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
        config_path.write_text(yaml.dump(config))

        result = runner.invoke(main, [
            "run", "-c", str(config_path), "--dry-run",
            "--filter", "metadata.category=tech",
        ])
        assert result.exit_code == 0
        assert "0" in result.output


class TestTimeoutOption:
    """Test the --timeout CLI option."""

    def test_timeout_accepted_in_help(self, runner):
        """--timeout appears in run help."""
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--timeout" in result.output

    def test_timeout_override_in_dry_run(self, tmp_path, runner):
        """--timeout overrides config timeout in dry-run."""
        dataset_path = tmp_path / "samples.jsonl"
        config_path = tmp_path / "config.yaml"

        with open(dataset_path, "w") as f:
            f.write(json.dumps({"query": "q", "context": [], "answer": "a"}) + "\n")

        config = {
            "judge": {"model": "gpt-4o", "timeout": 30},
            "evaluations": [
                {
                    "name": "Test",
                    "dataset": str(dataset_path),
                    "metrics": ["faithfulness"],
                }
            ],
        }
        config_path.write_text(yaml.dump(config))

        result = runner.invoke(main, [
            "run", "-c", str(config_path), "--dry-run",
            "--timeout", "120",
        ])
        assert result.exit_code == 0
        assert "120" in result.output
