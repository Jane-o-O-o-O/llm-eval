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
from llm_eval.regression import load_baseline, check_regression
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
@click.option("--parallel", "-p", type=int, default=None, help="Number of parallel evaluations.")
@click.option("--fail-on", "fail_on", default=None, help="Failure mode: 'threshold' (default) or 'regression'.")
@click.option("--baseline", "baseline_path", default=None, help="Baseline report JSON for regression comparison.")
@click.option("--tolerance", type=float, default=0.05, help="Regression tolerance (default: 0.05).")
def run(
    config_path: str,
    output_format: str | None,
    report_path: str | None,
    threshold: float | None,
    parallel: int | None,
    fail_on: str | None,
    baseline_path: str | None,
    tolerance: float,
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

    # Load baseline for regression mode
    baseline_scores: dict[str, float] | None = None
    if fail_on == "regression" and baseline_path:
        try:
            baseline_scores = load_baseline(baseline_path)
        except (FileNotFoundError, ValueError) as exc:
            click.echo(f"❌ Baseline error: {exc}", err=True)
            sys.exit(1)

    # Process each evaluation
    all_passed = True

    for eval_def in config.evaluations:
        eval_name = eval_def.get("name", "Unnamed")
        dataset_path = eval_def.get("dataset", "")
        metric_names = eval_def.get("metrics", [])
        eval_threshold = eval_def.get("threshold", config.threshold)
        eval_parallel = parallel or config.parallel

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
        click.echo(f"   ⚡ Parallel: {eval_parallel}")

        evaluator = Evaluator(
            metrics=metric_names,
            threshold=eval_threshold,
            parallel=eval_parallel,
        )

        # Progress bar
        pbar = _create_progress_bar(len(samples))

        def on_progress(current: int, total: int) -> None:
            if pbar:
                pbar.update(1)

        # Run evaluation
        try:
            results = asyncio.run(evaluator.evaluate(samples, on_progress=on_progress))
        except Exception as exc:
            if pbar:
                pbar.close()
            click.echo(f"❌ Evaluation failed: {exc}", err=True)
            sys.exit(1)
        finally:
            if pbar:
                pbar.close()

        summary = evaluator.summarize(results)

        # Output report
        report_content = _format_report(fmt, results, summary)

        if report_path:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            click.echo(f"   📄 Report saved to: {report_path}")

            # Also save as JSON for regression baseline
            if fmt != "json":
                json_report_path = f"{os.path.splitext(report_path)[0]}.json"
                json_content = format_json_report(results, summary)
                with open(json_report_path, "w", encoding="utf-8") as f:
                    f.write(json_content)
        else:
            click.echo(report_content)

        # Determine pass/fail
        if fail_on == "regression" and baseline_scores:
            reg_result = check_regression(results, baseline_scores, tolerance=tolerance)
            click.echo(reg_result.format_terminal())
            if not reg_result.passed:
                all_passed = False
                click.echo(f"\n❌ REGRESSION DETECTED: {eval_name}")
            else:
                click.echo(f"\n✅ NO REGRESSION: {eval_name}")
        else:
            # Default threshold mode
            if summary["overall_score"] < eval_threshold:
                click.echo(f"\n❌ FAILED: Score {summary['overall_score']:.2f} < threshold {eval_threshold}")
                all_passed = False
            else:
                click.echo(f"\n✅ PASSED: Score {summary['overall_score']:.2f} >= threshold {eval_threshold}")

    if not all_passed:
        sys.exit(1)


def _create_progress_bar(total: int):
    """Create a tqdm progress bar if tqdm is available."""
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc="Evaluating", unit="sample")
    except ImportError:
        return None


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
