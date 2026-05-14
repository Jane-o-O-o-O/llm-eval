"""Tests for CLI presets, multi-format output, and new CLI features."""

from __future__ import annotations

import json
import os

import yaml
from click.testing import CliRunner

from llm_eval.cli import PRESETS, main


class TestPresets:
    """Tests for config presets."""

    def test_presets_exist(self) -> None:
        assert "rag" in PRESETS
        assert "chatbot" in PRESETS
        assert "summarization" in PRESETS

    def test_preset_has_description(self) -> None:
        for name, data in PRESETS.items():
            assert "description" in data, f"Preset {name} missing description"
            assert data["description"], f"Preset {name} has empty description"

    def test_preset_config_is_valid_yaml(self) -> None:
        for name, data in PRESETS.items():
            config = yaml.safe_load(data["config"])
            assert "judge" in config, f"Preset {name} config missing 'judge'"
            assert "evaluations" in config, f"Preset {name} config missing 'evaluations'"

    def test_preset_samples_are_valid_jsonl(self) -> None:
        for name, data in PRESETS.items():
            lines = [ln for ln in data["samples"].strip().split("\n") if ln.strip()]
            assert len(lines) >= 1, f"Preset {name} has no sample lines"
            for line in lines:
                obj = json.loads(line)
                assert "query" in obj
                assert "answer" in obj


class TestInitWithPreset:
    """Tests for `llm-eval init --preset`."""

    def test_init_preset_rag(self, tmp_path) -> None:
        runner = CliRunner()
        output = str(tmp_path / "evals.yaml")
        result = runner.invoke(main, ["init", "--preset", "rag", "--output", output])
        assert result.exit_code == 0
        assert os.path.exists(output)
        with open(output) as f:
            config = yaml.safe_load(f)
        # RAG preset should have faithfulness
        metrics = config["evaluations"][0]["metrics"]
        assert "faithfulness" in metrics

    def test_init_preset_chatbot(self, tmp_path) -> None:
        runner = CliRunner()
        output = str(tmp_path / "evals.yaml")
        result = runner.invoke(main, ["init", "--preset", "chatbot", "--output", output])
        assert result.exit_code == 0
        with open(output) as f:
            config = yaml.safe_load(f)
        metrics = config["evaluations"][0]["metrics"]
        assert "coherence" in metrics
        assert "toxicity" in metrics

    def test_init_preset_summarization(self, tmp_path) -> None:
        runner = CliRunner()
        output = str(tmp_path / "evals.yaml")
        result = runner.invoke(main, ["init", "--preset", "summarization", "--output", output])
        assert result.exit_code == 0
        with open(output) as f:
            config = yaml.safe_load(f)
        metrics = config["evaluations"][0]["metrics"]
        assert "hallucination" in metrics

    def test_init_preset_creates_dataset(self, tmp_path) -> None:
        runner = CliRunner()
        output = str(tmp_path / "evals.yaml")
        runner.invoke(main, ["init", "--preset", "rag", "--output", output])
        dataset_path = tmp_path / "samples.jsonl"
        assert dataset_path.exists()

    def test_init_presets_command(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["presets"])
        assert result.exit_code == 0
        assert "rag" in result.output
        assert "chatbot" in result.output
        assert "summarization" in result.output


class TestMetricsVerbose:
    """Tests for `llm-eval metrics --verbose`."""

    def test_verbose_flag(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["metrics", "list", "--verbose"])
        assert result.exit_code == 0
        # Verbose shows descriptions on separate lines
        assert "faithfulness" in result.output
        # Verbose format has descriptions
        assert "Factual consistency" in result.output or "consistency" in result.output.lower()

    def test_metrics_normal_mode(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["metrics"])
        assert result.exit_code == 0
        assert "Total:" in result.output


class TestMultiFormatOutput:
    """Tests for comma-separated --output format."""

    def test_validate_accepts_comma_separated(self, tmp_path) -> None:
        """Validate should accept multi-format output config."""
        config = {
            "judge": {"model": "gpt-4o"},
            "defaults": {"output_format": "json,html"},
            "evaluations": [
                {
                    "name": "test",
                    "dataset": "samples.jsonl",
                    "metrics": ["faithfulness"],
                }
            ],
        }
        config_path = tmp_path / "evals.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Create dataset
        ds_path = tmp_path / "samples.jsonl"
        ds_path.write_text('{"query": "q", "context": ["c"], "answer": "a"}\n')

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(config_path)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower()


class TestInitHelp:
    """Tests for init --help showing preset option."""

    def test_init_help_shows_preset(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["init", "--help"])
        assert result.exit_code == 0
        assert "preset" in result.output.lower() or "--preset" in result.output
