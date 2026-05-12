"""CLI entry point for llm-eval."""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys

import click
import yaml

from llm_eval.compare import compare_reports, format_terminal_comparison, load_report
from llm_eval.dataset import load_jsonl
from llm_eval.evaluator import Evaluator
from llm_eval.metrics import get_default_registry
from llm_eval.models import EvalConfig, JudgeConfig
from llm_eval.plugins import load_custom_metrics
from llm_eval.regression import check_regression, load_baseline
from llm_eval.report import (
    format_csv_report,
    format_html_report,
    format_json_report,
    format_terminal_report,
)


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
@click.option("--fail-on", "fail_on", default=None, type=click.Choice(["threshold", "regression"]),
              help="Failure mode: 'threshold' (default) or 'regression' (compare to baseline).")
@click.option("--baseline", "baseline_path", default=None, help="Baseline JSON report for regression check.")
@click.option("--tolerance", type=float, default=0.05, help="Regression tolerance (default 0.05).")
@click.option("--parallel", "-p", type=int, default=None, help="Number of parallel evaluations.")
@click.option("--dry-run", is_flag=True, default=False, help="Validate config and print plan without running.")
@click.option("--sample", "-s", type=int, default=None, help="Randomly sample N items from dataset (quick testing).")
@click.option("--seed", type=int, default=None, help="Random seed for --sample (reproducible).")
def run(
    config_path: str,
    output_format: str | None,
    report_path: str | None,
    threshold: float | None,
    fail_on: str | None,
    baseline_path: str | None,
    tolerance: float,
    parallel: int | None,
    dry_run: bool,
    sample: int | None,
    seed: int | None,
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

    # Dry-run mode: validate and print plan
    if dry_run:
        click.echo("🔍 DRY RUN — Validation Report")
        click.echo("=" * 50)
        click.echo(f"   Judge model: {config.judge.model}")
        click.echo(f"   Threshold: {config.threshold}")
        click.echo(f"   Output format: {fmt}")
        if config.metric_weights:
            click.echo(f"   Metric weights: {config.metric_weights}")
        click.echo(f"\n   Evaluations ({len(config.evaluations)}):")
        registry = get_default_registry()
        for eval_def in config.evaluations:
            name = eval_def.get("name", "Unnamed")
            dataset = eval_def.get("dataset", "")
            metric_names = eval_def.get("metrics", [])
            eval_parallel = parallel or eval_def.get("parallel", 1)
            click.echo(f"\n   📋 {name}")
            click.echo(f"      Dataset: {dataset}")
            click.echo(f"      Metrics: {', '.join(metric_names)}")
            click.echo(f"      Parallel: {eval_parallel}")

            # Validate metrics
            unknown = [m for m in metric_names if not registry.is_registered(m)]
            if unknown:
                click.echo(f"      ❌ Unknown metrics: {', '.join(unknown)}")
            else:
                click.echo(f"      ✅ All metrics valid")

            # Validate dataset exists
            ds_path = dataset
            if not os.path.isabs(ds_path):
                config_dir = os.path.dirname(os.path.abspath(config_path))
                ds_path = os.path.join(config_dir, ds_path)
            if os.path.exists(ds_path):
                try:
                    samples = load_jsonl(ds_path)
                    click.echo(f"      ✅ Dataset: {len(samples)} samples")
                except ValueError as exc:
                    click.echo(f"      ❌ Dataset error: {exc}")
            else:
                click.echo(f"      ❌ Dataset not found: {dataset}")

        click.echo(f"\n✅ Dry run complete. Remove --dry-run to execute.")
        return

    # Load custom metrics if configured
    if config.custom_metrics:
        try:
            loaded = load_custom_metrics(get_default_registry(), config.custom_metrics)
            click.echo(f"🔌 Loaded {len(loaded)} custom metric(s): {', '.join(loaded)}")
        except (ImportError, AttributeError, TypeError) as exc:
            click.echo(f"❌ Failed to load custom metrics: {exc}", err=True)
            sys.exit(1)

    # Process each evaluation
    for eval_def in config.evaluations:
        eval_name = eval_def.get("name", "Unnamed")
        dataset_path = eval_def.get("dataset", "")
        metric_names = eval_def.get("metrics", [])
        eval_threshold = eval_def.get("threshold", config.threshold)
        eval_parallel = parallel or eval_def.get("parallel", config.judge.max_retries)

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

        # Sample subset for quick testing
        if sample is not None and sample < len(samples):
            rng = random.Random(seed)
            samples = rng.sample(samples, sample)
            click.echo(f"   🎲 Sampled {sample} items from dataset (seed={seed})")

        click.echo(f"   📊 Loaded {len(samples)} samples")
        click.echo(f"   📏 Metrics: {', '.join(metric_names)}")

        evaluator = Evaluator(
            metrics=metric_names,
            threshold=eval_threshold,
            parallel=eval_parallel or 1,
            metric_weights=config.metric_weights or eval_def.get("metric_weights", {}),
            judge_config=config.judge,
        )

        # Progress bar
        use_progress = len(samples) > 1 and sys.stderr.isatty()
        progress_bar = None
        if use_progress:
            try:
                from tqdm import tqdm
                progress_bar = tqdm(total=len(samples), desc="   Evaluating", unit="sample")
            except ImportError:
                progress_bar = None

        def _on_progress(completed: int, total: int) -> None:
            if progress_bar is not None:
                progress_bar.update(1)
            elif not use_progress:
                pass  # silent in non-tty mode

        # Run evaluation
        try:
            results = asyncio.run(evaluator.evaluate(samples, progress_callback=_on_progress))
        except Exception as exc:
            if progress_bar:
                progress_bar.close()
            click.echo(f"❌ Evaluation failed: {exc}", err=True)
            sys.exit(1)
        finally:
            if progress_bar:
                progress_bar.close()

        summary = evaluator.summarize(results)

        # Output report
        report_content = _format_report(fmt, results, summary)

        if report_path:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            click.echo(f"   📄 Report saved to: {report_path}")
        else:
            click.echo(report_content)

        # Failure mode
        if fail_on == "regression" and baseline_path:
            # Regression check
            try:
                baseline = load_baseline(baseline_path)
                reg_result = check_regression(results, baseline, tolerance=tolerance)
                click.echo(reg_result.format_terminal())
                if not reg_result.passed:
                    click.echo(f"\n❌ REGRESSION DETECTED (tolerance: {tolerance:.1%})")
                    sys.exit(1)
                else:
                    click.echo(f"\n✅ NO REGRESSION (tolerance: {tolerance:.1%})")
            except (FileNotFoundError, ValueError) as exc:
                click.echo(f"❌ Regression check failed: {exc}", err=True)
                sys.exit(1)
        else:
            # Default threshold check
            if summary["overall_score"] < eval_threshold:
                click.echo(f"\n❌ FAILED: Score {summary['overall_score']:.2f} < threshold {eval_threshold}")
                sys.exit(1)
            else:
                click.echo(f"\n✅ PASSED: Score {summary['overall_score']:.2f} >= threshold {eval_threshold}")


@main.command("metrics")
def metrics_list() -> None:
    """List all available evaluation metrics."""
    registry = get_default_registry()
    names = registry.list_metrics()

    click.echo("📋 Available Metrics\n")
    click.echo(f"{'Name':<22} {'Description'}")
    click.echo("─" * 60)
    for name in names:
        metric = registry.get(name)
        click.echo(f"{metric.name:<22} {metric.description}")
    click.echo(f"\nTotal: {len(names)} metrics")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
def validate(config_path: str) -> None:
    """Validate an evaluation config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # Check top-level structure
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(raw_config, dict):
        errors.append("Config must be a YAML mapping")
        click.echo(f"❌ Validation failed:\n  " + "\n  ".join(errors))
        sys.exit(1)

    # Check judge section
    judge_data = raw_config.get("judge", {})
    if not judge_data.get("model"):
        warnings.append("No judge model specified, will use default 'gpt-4o'")

    # Check evaluations
    evaluations = raw_config.get("evaluations", [])
    if not evaluations:
        errors.append("No evaluations defined")

    registry = get_default_registry()
    for i, eval_def in enumerate(evaluations):
        eval_name = eval_def.get("name", f"Evaluation #{i+1}")

        # Check metrics
        metric_names = eval_def.get("metrics", [])
        if not metric_names:
            errors.append(f"{eval_name}: No metrics specified")
        for m_name in metric_names:
            if not registry.is_registered(m_name):
                errors.append(f"{eval_name}: Unknown metric '{m_name}'")

        # Check dataset
        dataset = eval_def.get("dataset", "")
        if not dataset:
            errors.append(f"{eval_name}: No dataset path specified")
        else:
            if not os.path.isabs(dataset):
                config_dir = os.path.dirname(os.path.abspath(config_path))
                full_path = os.path.join(config_dir, dataset)
            else:
                full_path = dataset
            if not os.path.exists(full_path):
                errors.append(f"{eval_name}: Dataset not found: {dataset}")

        # Check threshold
        threshold = eval_def.get("threshold")
        if threshold is not None and (threshold < 0 or threshold > 1):
            errors.append(f"{eval_name}: Threshold must be between 0.0 and 1.0")

    # Check defaults
    defaults = raw_config.get("defaults", {})
    fmt = defaults.get("output_format", "terminal")
    if fmt not in ("terminal", "json", "csv", "html"):
        errors.append(f"Unknown output format: {fmt}")

    # Report
    if errors:
        click.echo(f"❌ Validation failed ({len(errors)} error(s)):\n")
        for err in errors:
            click.echo(f"  ❌ {err}")
        if warnings:
            click.echo("")
            for warn in warnings:
                click.echo(f"  ⚠️  {warn}")
        sys.exit(1)
    else:
        click.echo(f"✅ Config is valid!")
        if warnings:
            for warn in warnings:
                click.echo(f"  ⚠️  {warn}")
        click.echo(f"   {len(evaluations)} evaluation(s) defined")


@main.command()
@click.argument("report_a", type=click.Path(exists=True))
@click.argument("report_b", type=click.Path(exists=True))
@click.option("--label-a", default="Baseline", help="Label for the first report.")
@click.option("--label-b", default="Current", help="Label for the second report.")
@click.option("--output", "-o", "output_format", default="terminal", help="Output format: terminal, json.")
@click.option("--report", "-r", "report_path", default=None, help="Write comparison to file.")
def compare(
    report_a: str,
    report_b: str,
    label_a: str,
    label_b: str,
    output_format: str,
    report_path: str | None,
) -> None:
    """Compare two evaluation reports."""
    try:
        data_a = load_report(report_a)
        data_b = load_report(report_b)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"❌ Error loading report: {exc}", err=True)
        sys.exit(1)

    comparison = compare_reports(
        data_a, data_b, label_a=label_a, label_b=label_b,
        path_a=report_a, path_b=report_b,
    )

    if output_format == "json":
        content = json.dumps(comparison, indent=2, ensure_ascii=False)
    else:
        content = format_terminal_comparison(comparison)

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        click.echo(f"📄 Comparison saved to: {report_path}")
    else:
        click.echo(content)


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
