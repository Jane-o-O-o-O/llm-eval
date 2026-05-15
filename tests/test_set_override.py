"""Tests for --set CLI overrides on the run command."""

from __future__ import annotations

from unittest.mock import patch

from click.testing import CliRunner

from llm_eval.cli import main


class TestSetOverrides:
    """Test --set key=value config overrides."""

    def test_set_judge_model(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            "judge:\n  model: gpt-4o\n\n"
            "evaluations:\n"
            "  - name: test\n    dataset: data.jsonl\n    metrics: [faithfulness]\n"
        )
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"query": "q", "context": ["c"], "answer": "a"}\n')

        runner = CliRunner()
        with patch("llm_eval.cli.Evaluator") as mock_eval:
            mock_eval.return_value.evaluate = self._async_return([])
            mock_eval.return_value.summarize.return_value = {
                "total_samples": 0, "overall_score": 0.0,
                "pass_count": 0, "fail_count": 0, "pass_rate": 0.0,
                "metric_scores": {}, "median": 0, "p25": 0, "p75": 0,
                "std_dev": 0, "min_score": 0, "max_score": 0,
            }
            result = runner.invoke(main, [
                "run", "--config", str(config),
                "--set", "judge.model=claude-3-opus",
                "--dry-run",
            ])
            assert result.exit_code == 0
            assert "claude-3-opus" in result.output

    def test_set_multiple_overrides(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            "judge:\n  model: gpt-4o\n\n"
            "defaults:\n  threshold: 0.7\n\n"
            "evaluations:\n"
            "  - name: test\n    dataset: data.jsonl\n    metrics: [faithfulness]\n"
        )
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"query": "q", "context": ["c"], "answer": "a"}\n')

        runner = CliRunner()
        result = runner.invoke(main, [
            "run", "--config", str(config),
            "--set", "judge.model=claude-3-opus",
            "--set", "defaults.threshold=0.9",
            "--dry-run",
        ])
        assert result.exit_code == 0
        assert "claude-3-opus" in result.output
        assert "0.9" in result.output

    def test_set_invalid_format(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            "judge:\n  model: gpt-4o\n\n"
            "evaluations:\n"
            "  - name: test\n    dataset: data.jsonl\n    metrics: [faithfulness]\n"
        )
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"query": "q", "context": ["c"], "answer": "a"}\n')

        runner = CliRunner()
        result = runner.invoke(main, [
            "run", "--config", str(config),
            "--set", "invalid_no_equals",
        ])
        assert result.exit_code != 0
        assert "=" in result.output or "Invalid" in result.output

    def test_set_with_dot_notation(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text(
            "judge:\n  model: gpt-4o\n  temperature: 0\n\n"
            "evaluations:\n"
            "  - name: test\n    dataset: data.jsonl\n    metrics: [faithfulness]\n"
        )
        ds = tmp_path / "data.jsonl"
        ds.write_text('{"query": "q", "context": ["c"], "answer": "a"}\n')

        runner = CliRunner()
        result = runner.invoke(main, [
            "run", "--config", str(config),
            "--set", "judge.temperature=0.5",
            "--dry-run",
        ])
        assert result.exit_code == 0

    @staticmethod
    async def _async_return(val):
        return val
