"""CLI entry point for llm-eval."""

from __future__ import annotations

import asyncio
import json
import os
import sys

import click
import yaml

from llm_eval.dataset import load_jsonl
from llm_eval.evaluator import Evaluator
from llm_eval.models import EvalConfig
from llm_eval.report import format_csv_report, format_html_report, format_json_report, format_terminal_report


INIT_CONFIG_TEMPLATE = """\
judge:
  model: gpt-4o
  temperature: 0

defaults:
  threshold: 0.7
  parallel: 5
  output_format: terminal

evaluations:
  - name: "My RAG Pipeline"
    type: rag
    dataset: samples.jsonl
    metrics:
      - faithfulness
      - answer_relevancy
"""

INIT_SAMPLES_TEMPLATE = """\
{"query": "What is the refund policy?", "context": ["Refunds are processed within 5 business days."], "answer": "Refunds take up to 5 business days.", "reference": "Refunds are processed within 5 business days."}
{"query": "How do I reset my password?", "context": ["Click 'Forgot Password' on the login page."], "answer": "Go to the login page and click 'Forgot Password'.", "reference": "Navigate to login and click 'Forgot Password' to receive a reset email."}
"""


@click.group()
@click.version_option(package_name="llm-eval")
def main() -> None:
    """🧪 llm-eval — Lightweight CLI for evaluating LLM applications."""


@main.command()
@click.option("--output", "-o", default="evals.yaml", help="Output config file path.")
def init(output: str) -> None:
    """Initialize a new evaluation project with sample config and dataset."""
    click.echo(f"📝 Creating evaluation config: {output}")
    with open(output, "w", encoding="utf-8") as f:
        f.write(INIT_CONFIG_TEMPLATE)

    dataset_path = os.path.join(os.path.dirname(output) or ".", "samples.jsonl")
    click.echo(f"📦 Creating sample dataset: {dataset_path}")
    with open(dataset_path, "w", encoding="utf-8") as f:
        f.write(INIT_SAMPLES_TEMPLATE)

    click.echo("✅ Project initialized! Edit evals.yaml and samples.jsonl to get started.")
    click.echo("   Then run: llm-eval run --config evals.yaml")


@main.command()
@click.option("--config", "-c", "config_path", required=True, help="Path to eval config YAML.")
@click.option("--output", "-o", "output_format", default=None, help="Output format: terminal, json, csv, html.")
@click.option("--report", "-r", "report_path", default=None, help="Write report to file.")
@click.option("--threshold", "-t", type=float, default=None, help="Override pass/fail threshold.")
def run(
    config_path: str,
    output_format: str | None,
    report_path: str | None,
    threshold: float | None,
) -> None:
    """Run evaluations based on a config file."""
    # Load config
    if not os.path.exists(config_path):
        click.echo(f"❌ Config file not found: {config_path}", err=True)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    config = EvalConfig.from_dict(raw_config)
    if threshold is not None:
        config.threshold = threshold
    fmt = output_format or config.output_format

    # Process each evaluation
    for eval_def in config.evaluations:
        eval_name = eval_def.get("name", "Unnamed")
        dataset_path = eval_def.get("dataset", "")
        metric_names = eval_def.get("metrics", [])
        eval_threshold = eval_def.get("threshold", config.threshold)

        click.echo(f"\n🧪 Running evaluation: {eval_name}")

        # Resolve dataset path relative to config file
        if not os.path.isabs(dataset_path):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            dataset_path = os.path.join(config_dir, dataset_path)

        try:
            samples = load_jsonl(dataset_path)
        except (FileNotFoundError, ValueError) as exc:
            click.echo(f"❌ Dataset error: {exc}", err=True)
            sys.exit(1)

        click.echo(f"   📊 Loaded {len(samples)} samples")
        click.echo(f"   📏 Metrics: {', '.join(metric_names)}")

        evaluator = Evaluator(
            metrics=metric_names,
            threshold=eval_threshold,
        )

        # Run evaluation
        try:
            results = asyncio.run(evaluator.evaluate(samples))
        except Exception as exc:
            click.echo(f"❌ Evaluation failed: {exc}", err=True)
            sys.exit(1)

        summary = evaluator.summarize(results)

        # Output report
        report_content = _format_report(fmt, results, summary)

        if report_path:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            click.echo(f"   📄 Report saved to: {report_path}")
        else:
            click.echo(report_content)

        # Exit code based on threshold
        if summary["overall_score"] < eval_threshold:
            click.echo(f"\n❌ FAILED: Score {summary['overall_score']:.2f} < threshold {eval_threshold}")
            sys.exit(1)
        else:
            click.echo(f"\n✅ PASSED: Score {summary['overall_score']:.2f} >= threshold {eval_threshold}")


def _format_report(fmt: str, results, summary) -> str:
    """Format the report based on the specified format."""
    if fmt == "json":
        return format_json_report(results, summary)
    elif fmt == "csv":
        return format_csv_report(results)
    elif fmt == "html":
        return format_html_report(results, summary)
    else:
        return format_terminal_report(results, summary)


if __name__ == "__main__":
    main()
