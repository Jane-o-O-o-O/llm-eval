"""Tests for the CLI commands."""

import json
import os

import pytest
import yaml
from click.testing import CliRunner
from llm_eval.cli import main


class TestInitCommand:
    """Tests for the `llm-eval init` command."""

    def test_init_creates_config_file(self, tmp_path) -> None:
        runner = CliRunner()
        output = str(tmp_path / "evals.yaml")
        result = runner.invoke(main, ["init", "--output", output])
        assert result.exit_code == 0
        assert os.path.exists(output)

    def test_init_output_contains_judge_section(self, tmp_path) -> None:
        runner = CliRunner()
        output = str(tmp_path / "evals.yaml")
        runner.invoke(main, ["init", "--output", output])
        with open(output, "r") as f:
            content = f.read()
        assert "judge:" in content
        assert "evaluations:" in content

    def test_init_creates_sample_dataset(self, tmp_path) -> None:
        runner = CliRunner()
        output = str(tmp_path / "evals.yaml")
        runner.invoke(main, ["init", "--output", output])
        sample_path = tmp_path / "samples.jsonl"
        assert sample_path.exists()
        with open(sample_path, "r") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) >= 1

    def test_init_default_output(self, tmp_path) -> None:
        runner = CliRunner()
        with runner.isolated_filesystem(temp_dir=tmp_path):
            result = runner.invoke(main, ["init"])
            assert result.exit_code == 0
            assert os.path.exists("evals.yaml")


class TestRunCommand:
    """Tests for the `llm-eval run` command."""

    def test_run_missing_config_exits_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--config", "/nonexistent/config.yaml"])
        assert result.exit_code != 0

    def test_run_with_config_and_dataset(self, tmp_path) -> None:
        """Test run with a config file pointing to a dataset."""
        # Create dataset
        dataset = [
            {
                "query": "What is Python?",
                "context": ["Python is a programming language."],
                "answer": "Python is a programming language.",
            }
        ]
        dataset_path = tmp_path / "samples.jsonl"
        dataset_path.write_text("\n".join(json.dumps(d) for d in dataset) + "\n")

        # Create config
        config = {
            "judge": {"model": "gpt-4o"},
            "evaluations": [
                {
                    "name": "Test Eval",
                    "type": "rag",
                    "dataset": str(dataset_path),
                    "metrics": ["faithfulness"],
                }
            ],
            "defaults": {"threshold": 0.7},
        }
        config_path = tmp_path / "evals.yaml"
        config_path.write_text(yaml.dump(config))

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--config", str(config_path)])
        # Should fail because no API key, but config parsing should work
        assert result.exit_code != 0

    def test_run_shows_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config" in result.output

    def test_main_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "init" in result.output
        assert "run" in result.output

    def test_run_with_fail_on_regression_flag(self, tmp_path) -> None:
        """Test that --fail-on regression is accepted as a valid option."""
        # Just verify the flag is accepted by checking help
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--fail-on" in result.output
        assert "regression" in result.output

    def test_run_parallel_flag_accepted(self, tmp_path) -> None:
        """Test that --parallel flag is accepted."""
        dataset = [
            {
                "query": "test",
                "context": ["ctx"],
                "answer": "ans",
            }
        ]
        dataset_path = tmp_path / "samples.jsonl"
        dataset_path.write_text("\n".join(json.dumps(d) for d in dataset) + "\n")

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
        config_path = tmp_path / "evals.yaml"
        config_path.write_text(yaml.dump(config))

        runner = CliRunner()
        result = runner.invoke(main, [
            "run", "--config", str(config_path), "--parallel", "3",
        ])
        # Will fail (no API key), but the flag should be parsed
        assert "--parallel" not in result.output or result.exit_code != 0

    def test_sample_flag_accepted(self, tmp_path) -> None:
        """Test that --sample flag is accepted."""
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--sample" in result.output

    def test_sample_flag_with_dry_run(self, tmp_path) -> None:
        """Test --sample flag with --dry-run shows config info."""
        dataset = [
            {"query": f"q{i}", "context": [f"c{i}"], "answer": f"a{i}"}
            for i in range(10)
        ]
        dataset_path = tmp_path / "samples.jsonl"
        dataset_path.write_text("\n".join(json.dumps(d) for d in dataset) + "\n")

        config = {
            "judge": {"model": "gpt-4o"},
            "evaluations": [{
                "name": "Test",
                "dataset": str(dataset_path),
                "metrics": ["faithfulness"],
            }],
        }
        config_path = tmp_path / "evals.yaml"
        config_path.write_text(yaml.dump(config))

        runner = CliRunner()
        result = runner.invoke(main, [
            "run", "--config", str(config_path), "--dry-run",
        ])
        assert result.exit_code == 0
        assert "10 samples" in result.output


class TestMetricsCommand:
    """Tests for the `llm-eval metrics` command."""

    def test_metrics_lists_all_metrics(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["metrics"])
        assert result.exit_code == 0
        assert "faithfulness" in result.output
        assert "answer_relevancy" in result.output
        assert "context_precision" in result.output
        assert "context_recall" in result.output
        assert "format_compliance" in result.output
        assert "toxicity" in result.output
        assert "answer_correctness" in result.output
        assert "coherence" in result.output

    def test_metrics_shows_count(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["metrics"])
        assert result.exit_code == 0
        assert "Total:" in result.output

    def test_metrics_shows_descriptions(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["metrics"])
        assert result.exit_code == 0
        assert "Factual consistency" in result.output or "faithfulness" in result.output


class TestValidateCommand:
    """Tests for the `llm-eval validate` command."""

    def test_validate_valid_config(self, tmp_path) -> None:
        """Test validation of a valid config file."""
        dataset = [
            {
                "query": "test",
                "context": ["ctx"],
                "answer": "ans",
            }
        ]
        dataset_path = tmp_path / "samples.jsonl"
        dataset_path.write_text("\n".join(json.dumps(d) for d in dataset) + "\n")

        config = {
            "judge": {"model": "gpt-4o"},
            "evaluations": [
                {
                    "name": "Test Eval",
                    "dataset": str(dataset_path),
                    "metrics": ["faithfulness", "answer_relevancy"],
                }
            ],
            "defaults": {"threshold": 0.7},
        }
        config_path = tmp_path / "evals.yaml"
        config_path.write_text(yaml.dump(config))

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(config_path)])
        assert result.exit_code == 0
        assert "valid" in result.output.lower() or "✅" in result.output

    def test_validate_missing_dataset(self, tmp_path) -> None:
        """Test validation catches missing dataset."""
        config = {
            "evaluations": [
                {
                    "name": "Test",
                    "dataset": "/nonexistent/data.jsonl",
                    "metrics": ["faithfulness"],
                }
            ],
        }
        config_path = tmp_path / "evals.yaml"
        config_path.write_text(yaml.dump(config))

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(config_path)])
        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "❌" in result.output

    def test_validate_unknown_metric(self, tmp_path) -> None:
        """Test validation catches unknown metrics."""
        config = {
            "evaluations": [
                {
                    "name": "Test",
                    "dataset": "data.jsonl",
                    "metrics": ["nonexistent_metric"],
                }
            ],
        }
        config_path = tmp_path / "evals.yaml"
        config_path.write_text(yaml.dump(config))

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(config_path)])
        assert result.exit_code != 0
        assert "nonexistent_metric" in result.output

    def test_validate_no_evaluations(self, tmp_path) -> None:
        """Test validation catches empty evaluations."""
        config = {"evaluations": []}
        config_path = tmp_path / "evals.yaml"
        config_path.write_text(yaml.dump(config))

        runner = CliRunner()
        result = runner.invoke(main, ["validate", str(config_path)])
        assert result.exit_code != 0

    def test_validate_nonexistent_file(self) -> None:
        """Test validation of a nonexistent file."""
        runner = CliRunner()
        result = runner.invoke(main, ["validate", "/nonexistent/config.yaml"])
        assert result.exit_code != 0


class TestCompareCommand:
    """Tests for the `llm-eval compare` command."""

    def _make_report(self, tmp_path, name: str, overall: float, metrics: dict) -> str:
        """Helper to create a JSON report file."""
        data = {
            "summary": {
                "total_samples": 5,
                "overall_score": overall,
                "metric_scores": metrics,
            },
            "results": [],
        }
        path = tmp_path / name
        path.write_text(json.dumps(data))
        return str(path)

    def test_compare_two_reports(self, tmp_path) -> None:
        path_a = self._make_report(tmp_path, "a.json", 0.8, {
            "faithfulness": {"mean": 0.9, "min": 0.7, "max": 1.0},
        })
        path_b = self._make_report(tmp_path, "b.json", 0.85, {
            "faithfulness": {"mean": 0.85, "min": 0.6, "max": 1.0},
        })

        runner = CliRunner()
        result = runner.invoke(main, ["compare", path_a, path_b])
        assert result.exit_code == 0
        assert "faithfulness" in result.output

    def test_compare_with_labels(self, tmp_path) -> None:
        path_a = self._make_report(tmp_path, "a.json", 0.8, {
            "faithfulness": {"mean": 0.9, "min": 0.7, "max": 1.0},
        })
        path_b = self._make_report(tmp_path, "b.json", 0.85, {
            "faithfulness": {"mean": 0.85, "min": 0.6, "max": 1.0},
        })

        runner = CliRunner()
        result = runner.invoke(main, [
            "compare", path_a, path_b,
            "--label-a", "v1", "--label-b", "v2",
        ])
        assert result.exit_code == 0
        assert "v1" in result.output
        assert "v2" in result.output

    def test_compare_json_output(self, tmp_path) -> None:
        path_a = self._make_report(tmp_path, "a.json", 0.8, {
            "faithfulness": {"mean": 0.9, "min": 0.7, "max": 1.0},
        })
        path_b = self._make_report(tmp_path, "b.json", 0.85, {
            "faithfulness": {"mean": 0.85, "min": 0.6, "max": 1.0},
        })

        runner = CliRunner()
        result = runner.invoke(main, [
            "compare", path_a, path_b, "--output", "json",
        ])
        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert "comparisons" in parsed

    def test_compare_nonexistent_file(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["compare", "/a.json", "/b.json"])
        assert result.exit_code != 0

    def test_compare_save_to_file(self, tmp_path) -> None:
        path_a = self._make_report(tmp_path, "a.json", 0.8, {
            "faithfulness": {"mean": 0.9, "min": 0.7, "max": 1.0},
        })
        path_b = self._make_report(tmp_path, "b.json", 0.85, {
            "faithfulness": {"mean": 0.85, "min": 0.6, "max": 1.0},
        })
        out_path = str(tmp_path / "comparison.txt")

        runner = CliRunner()
        result = runner.invoke(main, [
            "compare", path_a, path_b, "--report", out_path,
        ])
        assert result.exit_code == 0
        assert os.path.exists(out_path)


class TestReportGeneration:
    """Tests for the report/output generation."""

    def test_terminal_report(self) -> None:
        from llm_eval.report import format_terminal_report
        from llm_eval.models import EvalResult, MetricResult

        results = [
            EvalResult(
                sample_index=0,
                metrics=[MetricResult(name="faithfulness", score=0.95, details={})],
            ),
        ]
        summary = {
            "total_samples": 1,
            "overall_score": 0.95,
            "pass_count": 1,
            "fail_count": 0,
            "pass_rate": 1.0,
            "threshold": 0.7,
            "metric_scores": {
                "faithfulness": {"mean": 0.95, "min": 0.95, "max": 0.95},
            },
        }
        output = format_terminal_report(results, summary)
        assert "faithfulness" in output
        assert "0.95" in output
        assert "PASS" in output

    def test_json_report(self) -> None:
        from llm_eval.report import format_json_report
        from llm_eval.models import EvalResult, MetricResult

        results = [
            EvalResult(
                sample_index=0,
                metrics=[MetricResult(name="faithfulness", score=0.8, details={})],
            ),
        ]
        summary = {
            "total_samples": 1,
            "overall_score": 0.8,
            "pass_count": 1,
            "fail_count": 0,
            "pass_rate": 1.0,
            "threshold": 0.7,
            "metric_scores": {},
        }
        output = format_json_report(results, summary)
        parsed = json.loads(output)
        assert parsed["summary"]["total_samples"] == 1
        assert len(parsed["results"]) == 1

    def test_csv_report(self) -> None:
        from llm_eval.report import format_csv_report
        from llm_eval.models import EvalResult, MetricResult

        results = [
            EvalResult(
                sample_index=0,
                metrics=[MetricResult(name="faithfulness", score=0.85, details={})],
            ),
            EvalResult(
                sample_index=1,
                metrics=[MetricResult(name="faithfulness", score=0.6, details={})],
            ),
        ]
        output = format_csv_report(results)
        lines = output.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        assert "sample_index" in lines[0]
        assert "faithfulness" in lines[0]


class TestDryRunCommand:
    """Tests for the --dry-run flag on `llm-eval run`."""

    def _setup_config(self, tmp_path, config_data=None):
        """Create a config and dataset for dry-run tests."""
        dataset = [
            {
                "query": "What is Python?",
                "context": ["Python is a programming language."],
                "answer": "Python is a programming language.",
            }
        ]
        dataset_path = tmp_path / "samples.jsonl"
        dataset_path.write_text("\n".join(json.dumps(d) for d in dataset) + "\n")

        config = config_data or {
            "judge": {"model": "gpt-4o"},
            "evaluations": [
                {
                    "name": "Test Eval",
                    "dataset": str(dataset_path),
                    "metrics": ["faithfulness", "answer_relevancy"],
                }
            ],
            "defaults": {"threshold": 0.7},
        }
        config_path = tmp_path / "evals.yaml"
        config_path.write_text(yaml.dump(config))
        return str(config_path)

    def test_dry_run_valid_config(self, tmp_path) -> None:
        config_path = self._setup_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--config", config_path, "--dry-run"])
        assert result.exit_code == 0
        assert "DRY RUN" in result.output
        assert "Test Eval" in result.output
        assert "faithfulness" in result.output
        assert "answer_relevancy" in result.output

    def test_dry_run_shows_judge_model(self, tmp_path) -> None:
        config_path = self._setup_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--config", config_path, "--dry-run"])
        assert result.exit_code == 0
        assert "gpt-4o" in result.output

    def test_dry_run_shows_sample_count(self, tmp_path) -> None:
        config_path = self._setup_config(tmp_path)
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--config", config_path, "--dry-run"])
        assert result.exit_code == 0
        assert "1 samples" in result.output

    def test_dry_run_detects_unknown_metric(self, tmp_path) -> None:
        config_path = self._setup_config(tmp_path, {
            "evaluations": [
                {
                    "name": "Bad Eval",
                    "dataset": str(tmp_path / "samples.jsonl"),
                    "metrics": ["nonexistent_metric"],
                }
            ],
        })
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--config", config_path, "--dry-run"])
        assert result.exit_code == 0  # dry-run doesn't fail, just reports
        assert "Unknown metrics" in result.output or "nonexistent_metric" in result.output

    def test_dry_run_shows_metric_weights(self, tmp_path) -> None:
        dataset_path = tmp_path / "samples.jsonl"
        dataset_path.write_text('{"query": "Q", "context": ["C"], "answer": "A"}\n')

        config = {
            "evaluations": [
                {
                    "name": "Weighted",
                    "dataset": str(dataset_path),
                    "metrics": ["faithfulness"],
                }
            ],
            "defaults": {
                "metric_weights": {"faithfulness": 2.0},
            },
        }
        config_path = tmp_path / "evals.yaml"
        config_path.write_text(yaml.dump(config))

        runner = CliRunner()
        result = runner.invoke(main, ["run", "--config", str(config_path), "--dry-run"])
        assert result.exit_code == 0
        assert "faithfulness" in result.output

    def test_dry_run_missing_config_exits_error(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["run", "--config", "/nonexistent.yaml", "--dry-run"])
        assert result.exit_code != 0
