"""CLI entry point for llm-eval."""

from __future__ import annotations

import asyncio
import json
import os
import random
import sys

import click
import yaml

from llm_eval.compare import (
    compare_reports,
    format_html_comparison,
    format_terminal_comparison,
    load_report,
)
from llm_eval.dataset import load_dataset
from llm_eval.evaluator import Evaluator
from llm_eval.history import list_runs, load_run, save_run
from llm_eval.markdown_report import format_markdown_report
from llm_eval.metrics import get_default_registry
from llm_eval.models import EvalConfig
from llm_eval.plugins import load_custom_metrics
from llm_eval.regression import check_regression, load_baseline
from llm_eval.report import (
    format_csv_report,
    format_html_report,
    format_json_report,
    format_junit_report,
    format_terminal_report,
    get_report_metadata,
)


def _echo(msg: str, quiet: bool = False) -> None:
    """Print message unless quiet mode is enabled."""
    if not quiet:
        click.echo(msg)


def _apply_filter(
    samples: list, filter_expr: str, quiet: bool = False
) -> list:
    """Filter samples by a metadata field expression.

    Supports expressions like ``metadata.category=tech`` which keeps only
    samples whose ``metadata["category"]`` equals ``"tech"``.

    Args:
        samples: List of Sample objects.
        filter_expr: Filter expression in ``field=value`` format.
        quiet: Suppress informational output.

    Returns:
        Filtered list of samples.

    Raises:
        click.BadParameter: If the expression format is invalid.
    """
    if "=" not in filter_expr:
        raise click.BadParameter(
            f"Invalid filter format: '{filter_expr}'. Use field=value (e.g. metadata.category=tech)."
        )

    field_path, _, value = filter_expr.partition("=")
    parts = field_path.split(".")

    filtered = []
    for sample in samples:
        obj: Any = sample
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif isinstance(obj, dict) and part in obj:
                obj = obj[part]
            else:
                obj = None
                break
        if str(obj) == value:
            filtered.append(sample)

    _echo(f"   🔍 Filtered: {len(filtered)}/{len(samples)} samples match '{filter_expr}'", quiet)
    return filtered


def _resolve_extends(raw_config: dict, config_dir: str) -> dict:
    """Resolve config file extends/inheritance.

    If the config has an 'extends' key, loads the base config and deep-merges.
    Child values override base values. Dicts are merged recursively;
    lists and scalars are replaced.

    Args:
        raw_config: The parsed child config dict.
        config_dir: Directory of the child config (for resolving relative paths).

    Returns:
        Merged config dict with 'extends' key removed.
    """
    extends = raw_config.get("extends")
    if not extends:
        return raw_config

    base_path = extends
    if not os.path.isabs(base_path):
        base_path = os.path.join(config_dir, base_path)

    if not os.path.exists(base_path):
        raise ValueError(f"Base config not found: {base_path}")

    with open(base_path, encoding="utf-8") as f:
        base_config = yaml.safe_load(f) or {}

    # Recursively resolve if base also extends
    base_dir = os.path.dirname(os.path.abspath(base_path))
    base_config = _resolve_extends(base_config, base_dir)

    # Deep merge: child overrides base
    merged = _deep_merge(base_config, raw_config)
    merged.pop("extends", None)
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dicts. Override values take precedence.

    Dicts are merged recursively. Lists and scalars are replaced.
    """
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


# ─── Config presets ───────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "rag": {
        "description": "RAG pipeline evaluation (faithfulness + relevancy + context)",
        "config": """\
judge:
  model: gpt-4o
  temperature: 0

defaults:
  threshold: 0.7
  parallel: 5
  output_format: terminal

evaluations:
  - name: "RAG Pipeline Evaluation"
    dataset: samples.jsonl
    metrics:
      - faithfulness
      - answer_relevancy
      - context_precision
      - context_recall
""",
        "samples": """\
{"query": "What is the refund policy?", "context": ["Refunds are processed within 5 business days."], "answer": "Refunds take up to 5 business days.", "reference": "Refunds are processed within 5 business days."}
{"query": "How do I reset my password?", "context": ["Click 'Forgot Password' on the login page."], "answer": "Go to the login page and click 'Forgot Password'.", "reference": "Navigate to login and click 'Forgot Password' to receive a reset email."}
""",
    },
    "chatbot": {
        "description": "Chatbot quality evaluation (coherence + relevancy + safety)",
        "config": """\
judge:
  model: gpt-4o
  temperature: 0

defaults:
  threshold: 0.7
  parallel: 5
  output_format: terminal

evaluations:
  - name: "Chatbot Quality Check"
    dataset: samples.jsonl
    metrics:
      - answer_relevancy
      - coherence
      - toxicity
""",
        "samples": """\
{"query": "Tell me a joke", "context": [], "answer": "Why did the chicken cross the road? To get to the other side!"}
{"query": "What's the capital of France?", "context": ["France is a country in Western Europe."], "answer": "The capital of France is Paris."}
""",
    },
    "summarization": {
        "description": "Summarization quality (faithfulness + hallucination + format)",
        "config": """\
judge:
  model: gpt-4o
  temperature: 0

defaults:
  threshold: 0.75
  parallel: 5
  output_format: terminal

evaluations:
  - name: "Summarization Quality"
    dataset: samples.jsonl
    metrics:
      - faithfulness
      - hallucination
      - answer_similarity
      - coherence
""",
        "samples": """\
{"query": "Summarize the following article", "context": ["The article discusses climate change effects on polar bears."], "answer": "Climate change is negatively impacting polar bear populations.", "reference": "Climate change is causing habitat loss for polar bears, leading to population decline."}
""",
    },
}


def _write_outputs(
    fmt: str,
    results: list,
    summary: dict,
    report_path: str | None,
    quiet: bool,
    metadata: dict | None = None,
) -> None:
    """Write report output in one or more formats.

    Supports comma-separated formats like 'json,html'.
    When multiple formats and a report_path are specified, appends the format
    extension to the base path (e.g., report.json, report.html).
    """
    formats = [f.strip() for f in fmt.split(",") if f.strip()]
    metadata = metadata or {}

    for single_fmt in formats:
        report_content = _format_report(single_fmt, results, summary, metadata)

        if report_path and len(formats) > 1:
            # Multi-format: append extension
            base, _ = os.path.splitext(report_path)
            actual_path = f"{base}.{single_fmt}" if single_fmt != "terminal" else report_path
            with open(actual_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            _echo(f"   📄 Report saved to: {actual_path}", quiet)
        elif report_path:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)
            _echo(f"   📄 Report saved to: {report_path}", quiet)
        elif not quiet:
            click.echo(report_content)


# ─── Main CLI group ───────────────────────────────────────────────────────

@click.group()
@click.version_option(package_name="llm-eval")
def main() -> None:
    """🧪 llm-eval — Lightweight CLI for evaluating LLM applications."""


@main.command()
@click.option("--output", "-o", default="evals.yaml", help="Output config file path.")
@click.option(
    "--preset",
    type=click.Choice(list(PRESETS.keys())),
    default=None,
    help="Use a preset config template (rag, chatbot, summarization).",
)
def init(output: str, preset: str | None) -> None:
    """Initialize a new evaluation project with sample config and dataset."""
    if preset:
        preset_data = PRESETS[preset]
        config_content = preset_data["config"]
        samples_content = preset_data["samples"]
        click.echo(f"📝 Creating {preset} evaluation config: {output}")
    else:
        config_content = PRESETS["rag"]["config"]
        samples_content = PRESETS["rag"]["samples"]
        click.echo(f"📝 Creating evaluation config: {output}")

    with open(output, "w", encoding="utf-8") as f:
        f.write(config_content)

    dataset_path = os.path.join(os.path.dirname(output) or ".", "samples.jsonl")
    click.echo(f"📦 Creating sample dataset: {dataset_path}")
    with open(dataset_path, "w", encoding="utf-8") as f:
        f.write(samples_content)

    click.echo("✅ Project initialized! Edit the config and dataset to get started.")
    click.echo(f"   Then run: llm-eval run --config {output}")


@main.command()
@click.option("--config", "-c", "config_path", required=True, help="Path to eval config YAML.")
@click.option(
    "--output",
    "-o",
    "output_format",
    default=None,
    help="Output format(s): terminal, json, csv, html (comma-separated for multiple).",
)
@click.option("--report", "-r", "report_path", default=None, help="Write report to file.")
@click.option("--threshold", "-t", type=float, default=None, help="Override pass/fail threshold.")
@click.option("--model", "-m", default=None, help="Override judge model (e.g. gpt-4o, claude-3-opus).")
@click.option(
    "--fail-on",
    "fail_on",
    default=None,
    type=click.Choice(["threshold", "regression"]),
    help="Failure mode: 'threshold' (default) or 'regression' (compare to baseline).",
)
@click.option(
    "--baseline", "baseline_path", default=None, help="Baseline JSON report for regression check."
)
@click.option("--tolerance", type=float, default=0.05, help="Regression tolerance (default 0.05).")
@click.option("--parallel", "-p", type=int, default=None, help="Number of parallel evaluations.")
@click.option(
    "--dry-run", is_flag=True, default=False, help="Validate config and print plan without running."
)
@click.option(
    "--sample",
    "-s",
    type=int,
    default=None,
    help="Randomly sample N items from dataset (quick testing).",
)
@click.option("--seed", type=int, default=None, help="Random seed for --sample (reproducible).")
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    default=False,
    help="Minimal output for CI/CD. Only prints pass/fail.",
)
@click.option("--tag", default=None, help="Tag this evaluation run for history tracking.")
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable judge response caching for this run.",
)
@click.option(
    "--save-history",
    is_flag=True,
    default=True,
    help="Save run to history (default: on).",
)
@click.option(
    "--filter",
    "filter_expr",
    default=None,
    help="Filter samples by metadata field (e.g. metadata.category=tech).",
)
@click.option(
    "--timeout",
    type=int,
    default=None,
    help="Override judge timeout in seconds for this run.",
)
def run(
    config_path: str,
    output_format: str | None,
    report_path: str | None,
    threshold: float | None,
    model: str | None,
    fail_on: str | None,
    baseline_path: str | None,
    tolerance: float,
    parallel: int | None,
    dry_run: bool,
    sample: int | None,
    seed: int | None,
    quiet: bool,
    tag: str | None,
    no_cache: bool,
    save_history: bool,
    filter_expr: str | None,
    timeout: int | None,
) -> None:
    """Run evaluations based on a config file."""
    # Load config
    if not os.path.exists(config_path):
        click.echo(f"❌ Config file not found: {config_path}", err=True)
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # Resolve extends/inheritance
    config_dir = os.path.dirname(os.path.abspath(config_path))
    raw_config = _resolve_extends(raw_config, config_dir)

    config = EvalConfig.from_dict(raw_config)
    if threshold is not None:
        config.threshold = threshold
    if model is not None:
        config.judge.model = model
    if timeout is not None:
        config.judge.timeout = timeout
    fmt = output_format or config.output_format

    # Build report metadata
    metadata = get_report_metadata(config_path=config_path)

    # Dry-run mode: validate and print plan
    if dry_run:
        # Validate filter expression early
        if filter_expr and "=" not in filter_expr:
            click.echo(f"❌ Invalid filter format: '{filter_expr}'. Use field=value (e.g. metadata.category=tech)", err=True)
            sys.exit(1)

        click.echo("🔍 DRY RUN — Validation Report")
        click.echo("=" * 50)
        click.echo(f"   Judge model: {config.judge.model}")
        click.echo(f"   Threshold: {config.threshold}")
        click.echo(f"   Output format: {fmt}")
        click.echo(f"   Timeout: {config.judge.timeout}s")
        if timeout is not None:
            click.echo(f"   ⏱️  Timeout overridden via CLI: {timeout}s")
        if filter_expr:
            click.echo(f"   🔍 Filter: {filter_expr}")
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
                click.echo("      ✅ All metrics valid")

            # Validate dataset exists
            ds_path = dataset
            if not os.path.isabs(ds_path):
                config_dir = os.path.dirname(os.path.abspath(config_path))
                ds_path = os.path.join(config_dir, ds_path)
            if os.path.exists(ds_path):
                try:
                    samples = load_dataset(ds_path)
                    if filter_expr:
                        samples = _apply_filter(samples, filter_expr, quiet)
                    click.echo(f"      ✅ Dataset: {len(samples)} samples")
                except ValueError as exc:
                    click.echo(f"      ❌ Dataset error: {exc}")
            else:
                click.echo(f"      ❌ Dataset not found: {dataset}")

        click.echo("\n✅ Dry run complete. Remove --dry-run to execute.")
        return

    # Load custom metrics if configured
    if config.custom_metrics:
        try:
            loaded = load_custom_metrics(get_default_registry(), config.custom_metrics)
            _echo(f"🔌 Loaded {len(loaded)} custom metric(s): {', '.join(loaded)}", quiet)
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
        eval_use_cache = not no_cache
        eval_metric_options = eval_def.get("metric_options", {})

        _echo(f"\n🧪 Running evaluation: {eval_name}", quiet)

        # Resolve dataset path relative to config file
        if not os.path.isabs(dataset_path):
            config_dir = os.path.dirname(os.path.abspath(config_path))
            dataset_path = os.path.join(config_dir, dataset_path)

        try:
            samples = load_dataset(dataset_path)
        except (FileNotFoundError, ValueError) as exc:
            click.echo(f"❌ Dataset error: {exc}", err=True)
            sys.exit(1)

        # Apply filter
        if filter_expr:
            samples = _apply_filter(samples, filter_expr, quiet)

        # Sample subset for quick testing
        if sample is not None and sample < len(samples):
            rng = random.Random(seed)
            samples = rng.sample(samples, sample)
            _echo(f"   🎲 Sampled {sample} items from dataset (seed={seed})", quiet)

        _echo(f"   📊 Loaded {len(samples)} samples", quiet)
        _echo(f"   📏 Metrics: {', '.join(metric_names)}", quiet)

        evaluator = Evaluator(
            metrics=metric_names,
            threshold=eval_threshold,
            parallel=eval_parallel or 1,
            metric_weights=config.metric_weights or eval_def.get("metric_weights", {}),
            judge_config=config.judge,
            use_cache=eval_use_cache,
            metric_options=eval_metric_options,
        )

        # Progress bar
        use_progress = len(samples) > 1 and sys.stderr.isatty() and not quiet
        progress_bar = None
        if use_progress:
            try:
                from tqdm import tqdm

                progress_bar = tqdm(total=len(samples), desc="   Evaluating", unit="sample")
            except ImportError:
                progress_bar = None

        def _on_progress(completed: int, total: int, _pb=progress_bar, _up=use_progress) -> None:
            if _pb is not None:
                _pb.update(1)
            elif not _up:
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

        # Output report (supports multi-format)
        _write_outputs(fmt, results, summary, report_path, quiet, metadata)

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
                click.echo(
                    f"\n❌ FAILED: Score {summary['overall_score']:.2f} < threshold {eval_threshold}"
                )
                # Save to history even on failure
                if save_history:
                    hist_path = save_run(
                        results, summary, tag=tag, config_path=config_path
                    )
                    _echo(f"   📂 Run saved to: {hist_path}", quiet)
                sys.exit(1)
            else:
                click.echo(
                    f"\n✅ PASSED: Score {summary['overall_score']:.2f} >= threshold {eval_threshold}"
                )
                # Save to history on success
                if save_history:
                    hist_path = save_run(
                        results, summary, tag=tag, config_path=config_path
                    )
                    _echo(f"   📂 Run saved to: {hist_path}", quiet)


@main.group("metrics", invoke_without_command=True)
@click.pass_context
def metrics_group(ctx: click.Context) -> None:
    """List and manage evaluation metrics."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(metrics_list)


@metrics_group.command("list")
@click.option(
    "--verbose", "-v", is_flag=True, default=False, help="Show detailed metric information."
)
def metrics_list(verbose: bool) -> None:
    """List all available evaluation metrics."""
    registry = get_default_registry()
    names = registry.list_metrics()

    click.echo("📋 Available Metrics\n")
    if verbose:
        for name in names:
            metric = registry.get(name)
            click.echo(f"  🔹 {name}")
            click.echo(f"     {metric.description}")
            click.echo("")
    else:
        click.echo(f"{'Name':<22} {'Description'}")
        click.echo("─" * 60)
        for name in names:
            metric = registry.get(name)
            click.echo(f"{metric.name:<22} {metric.description}")
    click.echo(f"\nTotal: {len(names)} metrics")


@metrics_group.command("create")
@click.argument("name")
@click.option("--output", "-o", default=None, help="Output file path (default: {name}_metric.py).")
def metrics_create(name: str, output: str | None) -> None:
    """Scaffold a custom metric template file.

    Generates a Python file with a Metric subclass skeleton that you can
    fill in with your custom evaluation logic.

    Example: llm-eval metrics create my_custom_metric
    """
    class_name = "".join(word.capitalize() for word in name.split("_")) + "Metric"
    file_path = output or f"{name}_metric.py"

    template = f'''"""Custom metric: {name}

Generated by llm-eval metrics create.
Fill in the evaluate() method with your custom evaluation logic.
"""

from __future__ import annotations

from llm_eval.metrics import Metric, MetricResult
from llm_eval.models import JudgeConfig, Sample


class {class_name}(Metric):
    """Custom evaluation metric: {name}.

    Replace this docstring with a description of what this metric measures.
    """

    name = "{name}"
    description = "Custom metric: {name} (edit description)"

    def __init__(self, judge_config: JudgeConfig | None = None) -> None:
        super().__init__(judge_config=judge_config)

    async def evaluate(self, sample: Sample) -> MetricResult:
        """Evaluate a single sample.

        Args:
            sample: The evaluation sample containing query, context, answer.

        Returns:
            MetricResult with a score between 0.0 and 1.0.

        Implementation options:
        1. Use self._judge_call(prompt) to call the LLM judge
        2. Implement custom logic (regex, heuristics, etc.)
        3. Combine both approaches
        """
        # Option 1: LLM judge
        # prompt = f"Evaluate the following: query={{sample.query}} answer={{sample.answer}}"
        # result = await self._judge_call(prompt)
        # score = float(result.get("score", 0.0))

        # Option 2: Custom logic
        score = 1.0  # Replace with actual evaluation

        return MetricResult(
            name=self.name,
            score=min(max(score, 0.0), 1.0),  # Clamp to [0, 1]
            details={{"method": "custom"}},  # Add reasoning/details here
        )
'''

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(template)

    click.echo(f"📝 Created custom metric template: {file_path}")
    click.echo(f"   Class: {class_name}")
    click.echo(f"\n   Next steps:")
    click.echo(f"   1. Edit {file_path} and implement evaluate()")
    click.echo(f"   2. Add to your config:")
    click.echo(f"      custom_metrics:")
    click.echo(f"        - module: {os.path.splitext(os.path.basename(file_path))[0]}")
    click.echo(f"          class_name: {class_name}")


@main.command()
@click.argument("config_path", type=click.Path(exists=True))
def validate(config_path: str) -> None:
    """Validate an evaluation config file."""
    with open(config_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    # Check top-level structure
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(raw_config, dict):
        errors.append("Config must be a YAML mapping")
        click.echo("❌ Validation failed:\n  " + "\n  ".join(errors))
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
        eval_name = eval_def.get("name", f"Evaluation #{i + 1}")

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
    valid_formats = {"terminal", "json", "csv", "html", "markdown", "junit"}
    for f in fmt.split(","):
        if f.strip() not in valid_formats:
            errors.append(f"Unknown output format: {f.strip()}")

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
        click.echo("✅ Config is valid!")
        if warnings:
            for warn in warnings:
                click.echo(f"  ⚠️  {warn}")
        click.echo(f"   {len(evaluations)} evaluation(s) defined")


@main.command()
def doctor() -> None:
    """Check environment and configuration for common issues."""
    from llm_eval.cache import JudgeCache
    from llm_eval.metrics import get_default_registry

    click.echo("🩺 llm-eval Doctor\n")
    all_ok = True

    # 1. Python version
    import sys
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        click.echo(f"  ✅ Python {py_ver}")
    else:
        click.echo(f"  ❌ Python {py_ver} (requires >= 3.10)")
        all_ok = False

    # 2. Dependencies (import_name -> package_name for pip)
    deps = {"click": "click", "httpx": "httpx", "yaml": "PyYAML", "pydantic": "pydantic", "tqdm": "tqdm"}
    for import_name, pkg_name in deps.items():
        try:
            __import__(import_name)
            import importlib.metadata
            try:
                ver = importlib.metadata.version(pkg_name)
            except importlib.metadata.PackageNotFoundError:
                ver = "?"
            click.echo(f"  ✅ {import_name} ({ver})")
        except ImportError:
            click.echo(f"  ❌ {import_name} not installed")
            all_ok = False

    # 3. API keys
    click.echo("")
    keys = {
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
        "ANTHROPIC_API_KEY": os.environ.get("ANTHROPIC_API_KEY"),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY"),
    }
    found_any_key = False
    for key_name, key_val in keys.items():
        if key_val:
            masked = key_val[:8] + "..." + key_val[-4:] if len(key_val) > 12 else "***"
            click.echo(f"  ✅ {key_name} = {masked}")
            found_any_key = True
        else:
            click.echo(f"  ⚠️  {key_name} not set")
    if not found_any_key:
        click.echo("  💡 No API keys found. Set at least OPENAI_API_KEY to run evaluations.")

    # 4. Metrics
    click.echo("")
    registry = get_default_registry()
    metric_names = registry.list_metrics()
    click.echo(f"  ✅ {len(metric_names)} metrics available: {', '.join(metric_names)}")

    # 5. Cache
    click.echo("")
    try:
        cache = JudgeCache()
        stats = cache.stats()
        size_kb = stats["db_size_bytes"] / 1024
        click.echo(f"  ✅ Cache: {stats['entry_count']} entries ({size_kb:.1f} KB)")
        click.echo(f"     Path: {stats['db_path']}")
        cache.close()
    except Exception as exc:
        click.echo(f"  ⚠️  Cache error: {exc}")

    # 6. Version
    from llm_eval import __version__
    click.echo(f"\n  📦 llm-eval v{__version__}")

    if all_ok:
        click.echo("\n✅ All checks passed!")
    else:
        click.echo("\n❌ Some checks failed. Fix issues above.")


@main.command()
@click.argument("report_a", type=click.Path(exists=True))
@click.argument("report_b", type=click.Path(exists=True))
@click.option("--label-a", default="Baseline", help="Label for the first report.")
@click.option("--label-b", default="Current", help="Label for the second report.")
@click.option(
    "--output",
    "-o",
    "output_format",
    default="terminal",
    help="Output format: terminal, json, html.",
)
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
        data_a,
        data_b,
        label_a=label_a,
        label_b=label_b,
        path_a=report_a,
        path_b=report_b,
    )

    if output_format == "json":
        content = json.dumps(comparison, indent=2, ensure_ascii=False)
    elif output_format == "html":
        content = format_html_comparison(comparison)
    else:
        content = format_terminal_comparison(comparison)

    if report_path:
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(content)
        click.echo(f"📄 Comparison saved to: {report_path}")
    else:
        click.echo(content)


@main.command("presets")
def list_presets() -> None:
    """List available config presets."""
    click.echo("📋 Available Presets\n")
    for name, data in PRESETS.items():
        click.echo(f"  🔹 {name}")
        click.echo(f"     {data['description']}")
        click.echo("")
    click.echo("Usage: llm-eval init --preset rag")


# ─── Dataset subcommand group ────────────────────────────────────────────


@main.group()
def dataset() -> None:
    """Inspect and manage evaluation datasets."""


@dataset.command("info")
@click.argument("path", type=click.Path(exists=True))
def dataset_info(path: str) -> None:
    """Show dataset statistics and sample count."""
    try:
        samples = load_dataset(path)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"❌ Error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"📊 Dataset: {path}")
    click.echo(f"   Samples: {len(samples)}")

    # Field analysis
    has_reference = sum(1 for s in samples if s.reference)
    has_context = sum(1 for s in samples if s.context)
    avg_context = sum(len(s.context) for s in samples) / len(samples) if samples else 0
    avg_query_len = sum(len(s.query) for s in samples) / len(samples) if samples else 0
    avg_answer_len = sum(len(s.answer) for s in samples) / len(samples) if samples else 0

    click.echo(f"   With reference: {has_reference}/{len(samples)}")
    click.echo(f"   With context: {has_context}/{len(samples)}")
    click.echo(f"   Avg context chunks: {avg_context:.1f}")
    click.echo(f"   Avg query length: {avg_query_len:.0f} chars")
    click.echo(f"   Avg answer length: {avg_answer_len:.0f} chars")

    # Show first few samples
    click.echo("\n📋 First sample:")
    s = samples[0]
    click.echo(f"   Query: {s.query[:100]}{'...' if len(s.query) > 100 else ''}")
    click.echo(f"   Answer: {s.answer[:100]}{'...' if len(s.answer) > 100 else ''}")
    if s.context:
        click.echo(f"   Context: {len(s.context)} chunk(s)")


@dataset.command("validate")
@click.argument("path", type=click.Path(exists=True))
def dataset_validate(path: str) -> None:
    """Validate a dataset file for common issues."""
    errors: list[str] = []
    warnings: list[str] = []

    try:
        samples = load_dataset(path)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"❌ Dataset error: {exc}", err=True)
        sys.exit(1)

    for i, sample in enumerate(samples):
        idx = i + 1
        if not sample.query.strip():
            errors.append(f"Sample #{idx}: Empty query")
        if not sample.answer.strip():
            errors.append(f"Sample #{idx}: Empty answer")
        if not sample.context:
            warnings.append(f"Sample #{idx}: No context (ok for non-RAG tasks)")
        if sample.query == sample.answer:
            warnings.append(f"Sample #{idx}: Query identical to answer")

    if errors:
        click.echo(f"❌ Validation failed ({len(errors)} error(s)):\n")
        for err in errors:
            click.echo(f"  ❌ {err}")
        sys.exit(1)
    else:
        click.echo(f"✅ Dataset is valid! ({len(samples)} samples)")
        if warnings:
            for w in warnings:
                click.echo(f"  ⚠️  {w}")


@dataset.command("sample")
@click.argument("path", type=click.Path(exists=True))
@click.option("-n", "--count", type=int, default=3, help="Number of samples to show.")
@click.option("--seed", type=int, default=None, help="Random seed for reproducibility.")
def dataset_sample(path: str, count: int, seed: int | None) -> None:
    """Show random samples from a dataset."""
    try:
        samples = load_dataset(path)
    except (FileNotFoundError, ValueError) as exc:
        click.echo(f"❌ Error: {exc}", err=True)
        sys.exit(1)

    rng = random.Random(seed)
    selected = rng.sample(samples, min(count, len(samples)))

    for i, s in enumerate(selected, 1):
        click.echo(f"\n{'='*60}")
        click.echo(f"Sample #{i}")
        click.echo(f"{'='*60}")
        click.echo(f"Query:    {s.query}")
        click.echo(f"Answer:   {s.answer}")
        if s.context:
            for j, ctx in enumerate(s.context):
                click.echo(f"Context[{j}]: {ctx[:200]}{'...' if len(ctx) > 200 else ''}")
        if s.reference:
            click.echo(f"Reference: {s.reference[:200]}{'...' if len(s.reference) > 200 else ''}")


@dataset.command("convert")
@click.argument("source", type=click.Path(exists=True))
@click.argument("dest", type=click.Path())
def dataset_convert(source: str, dest: str) -> None:
    """Convert a dataset between JSONL and CSV formats.

    Format is auto-detected from file extension.
    """
    from llm_eval.dataset import load_csv, load_jsonl

    src_ext = os.path.splitext(source)[1].lower()
    dst_ext = os.path.splitext(dest)[1].lower()

    # Load
    if src_ext == ".csv":
        samples = load_csv(source)
    else:
        samples = load_jsonl(source)

    # Write
    if dst_ext == ".csv":
        import csv as csv_mod
        import io as io_mod

        output = io_mod.StringIO()
        writer = csv_mod.writer(output)
        writer.writerow(["query", "context", "answer", "reference"])
        for s in samples:
            ctx = json.dumps(s.context, ensure_ascii=False) if s.context else ""
            writer.writerow([s.query, ctx, s.answer, s.reference or ""])
        with open(dest, "w", encoding="utf-8", newline="") as f:
            f.write(output.getvalue())
    else:
        with open(dest, "w", encoding="utf-8") as f:
            for s in samples:
                row: dict[str, Any] = {
                    "query": s.query,
                    "context": s.context,
                    "answer": s.answer,
                }
                if s.reference:
                    row["reference"] = s.reference
                if s.metadata:
                    row["metadata"] = s.metadata
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    click.echo(f"✅ Converted {len(samples)} samples: {source} → {dest}")


# ─── History command ─────────────────────────────────────────────────────


@main.group(invoke_without_command=True)
@click.pass_context
@click.option("--tag", default=None, help="Filter runs by tag.")
@click.option("-n", "--limit", type=int, default=10, help="Number of runs to show.")
@click.option("--details", is_flag=True, default=False, help="Show per-metric details.")
def history(ctx: click.Context, tag: str | None, limit: int, details: bool) -> None:
    """Browse past evaluation runs."""
    if ctx.invoked_subcommand is not None:
        return

    runs = list_runs(tag=tag, limit=limit)

    if not runs:
        click.echo("📭 No evaluation runs found.")
        return

    click.echo(f"📂 Evaluation History ({len(runs)} runs)\n")
    click.echo(f"{'Timestamp':<22} {'Tag':<15} {'Score':<10} {'Samples':<8} {'File'}")
    click.echo("─" * 80)
    for r in runs:
        tag_str = r["tag"] or "—"
        ts = r["timestamp"][:19].replace("T", " ") if r["timestamp"] else "?"
        click.echo(
            f"{ts:<22} {tag_str:<15} {r['overall_score']:<10.4f} "
            f"{r['total_samples']:<8} {r['file']}"
        )

    if details and runs:
        click.echo(f"\n📊 Details for latest run: {runs[0]['file']}")
        run_data = load_run(runs[0]["path"])
        summary = run_data.get("summary", {})
        for metric_name, scores in summary.get("metric_scores", {}).items():
            click.echo(f"   {metric_name}: mean={scores['mean']:.4f}")


@history.command("trend")
@click.option("--tag", default=None, help="Filter runs by tag.")
@click.option("-n", "--limit", type=int, default=20, help="Number of runs to show.")
@click.option("--history-dir", default=None, help="Override history directory.")
def history_trend(tag: str | None, limit: int, history_dir: str | None) -> None:
    """Show score trends across evaluation runs.

    Displays a simple ASCII sparkline chart of scores over time.
    """
    runs = list_runs(tag=tag, limit=limit, history_dir=history_dir)

    if not runs:
        click.echo("📭 No evaluation runs found.")
        return

    # Reverse to chronological order
    runs = list(reversed(runs))
    scores = [r["overall_score"] for r in runs]

    click.echo(f"📈 Score Trend ({len(runs)} runs)\n")

    # Sparkline chart
    blocks = " ▁▂▃▄▅▆▇█"
    min_s = min(scores)
    max_s = max(scores)
    range_s = max_s - min_s if max_s > min_s else 1.0

    sparkline = ""
    for s in scores:
        idx = int((s - min_s) / range_s * (len(blocks) - 1))
        sparkline += blocks[idx]

    click.echo(f"  Score range: {min_s:.4f} — {max_s:.4f}")
    click.echo(f"  Latest:      {scores[-1]:.4f}")
    click.echo(f"  Mean:        {sum(scores)/len(scores):.4f}")
    click.echo(f"\n  {sparkline}")
    click.echo(f"  {'^' * len(sparkline)}")
    pad = max(0, len(sparkline) - 12)
    click.echo(f"  {'oldest':>6}{' ' * pad}{'newest':>6}")

    # Table of runs
    click.echo(f"\n{'#':<4} {'Timestamp':<22} {'Tag':<15} {'Score':<10} {'Samples'}")
    click.echo("─" * 60)
    for i, r in enumerate(runs, 1):
        ts = r["timestamp"][:19].replace("T", " ") if r["timestamp"] else "?"
        tag_str = r["tag"] or "—"
        click.echo(f"{i:<4} {ts:<22} {tag_str:<15} {r['overall_score']:<10.4f} {r['total_samples']}")


def _format_report(fmt: str, results, summary, metadata: dict | None = None) -> str:
    """Format the report based on the specified format."""
    metadata = metadata or {}
    if fmt == "json":
        return format_json_report(results, summary, metadata)
    elif fmt == "csv":
        return format_csv_report(results, metadata)
    elif fmt == "html":
        return format_html_report(results, summary, metadata)
    elif fmt == "markdown":
        return format_markdown_report(results, summary, metadata)
    elif fmt == "junit":
        return format_junit_report(results, summary, metadata)
    else:
        return format_terminal_report(results, summary, metadata)


if __name__ == "__main__":
    main()


# ─── Cache management commands ───────────────────────────────────────────


@main.group("cache", invoke_without_command=True)
@click.pass_context
def cache_group(ctx: click.Context) -> None:
    """Manage the judge response cache."""
    if ctx.invoked_subcommand is None:
        ctx.invoke(cache_stats)


@cache_group.command("stats")
@click.option(
    "--output", "-o", "output_format", default="terminal",
    help="Output format: terminal or json.",
)
def cache_stats(output_format: str) -> None:
    """Show cache statistics (entry count, size, path)."""
    from llm_eval.cache import JudgeCache

    try:
        cache = JudgeCache()
        stats = cache.stats()
        cache.close()
    except Exception as exc:
        click.echo(f"❌ Cache error: {exc}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(stats, indent=2))
    else:
        size_kb = stats["db_size_bytes"] / 1024
        click.echo("💾 Cache Statistics\n")
        click.echo(f"   Entries:  {stats['entry_count']}")
        click.echo(f"   Size:     {size_kb:.1f} KB")
        click.echo(f"   Path:     {stats['db_path']}")


@cache_group.command("clear")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
def cache_clear(yes: bool) -> None:
    """Remove all cached judge responses."""
    from llm_eval.cache import JudgeCache

    if not yes:
        click.confirm("⚠️  This will remove all cached responses. Continue?", abort=True)

    try:
        cache = JudgeCache()
        removed = cache.clear()
        cache.close()
    except Exception as exc:
        click.echo(f"❌ Cache error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"✅ Removed {removed} cached entries.")


@cache_group.command("purge")
@click.option("--older-than", type=int, required=True, help="Remove entries older than N days.")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation prompt.")
def cache_purge(older_than: int, yes: bool) -> None:
    """Remove cached entries older than N days."""
    from llm_eval.cache import JudgeCache

    if not yes:
        click.confirm(
            f"⚠️  This will remove cache entries older than {older_than} days. Continue?",
            abort=True,
        )

    try:
        cache = JudgeCache()
        # Purge by deleting entries older than N days
        import sqlite3 as _sqlite3
        conn = cache._conn
        count_before = conn.execute("SELECT COUNT(*) FROM judge_cache").fetchone()[0]
        conn.execute(
            "DELETE FROM judge_cache WHERE created_at < datetime('now', ?)",
            (f"-{older_than} days",),
        )
        conn.commit()
        count_after = conn.execute("SELECT COUNT(*) FROM judge_cache").fetchone()[0]
        removed = count_before - count_after
        cache.close()
    except Exception as exc:
        click.echo(f"❌ Cache error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"✅ Purged {removed} entries older than {older_than} days.")


# ─── Export command ──────────────────────────────────────────────────────


_FORMAT_EXTENSIONS = {
    ".json": "json",
    ".html": "html",
    ".htm": "html",
    ".csv": "csv",
    ".xml": "junit",
    ".md": "markdown",
}


@main.command()
@click.argument("input_path", type=click.Path(exists=True))
@click.option("--to", "to_format", default=None, help="Target format: json, html, csv, markdown, junit, terminal.")
@click.option("--output", "-o", "output_path", default=None, help="Output file path (stdout if omitted).")
def export(input_path: str, to_format: str | None, output_path: str | None) -> None:
    """Convert a JSON evaluation report to another format.

    Example: llm-eval export report.json --to html -o report.html
    """
    # Load JSON report
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    results_data = data.get("results", [])
    summary = data.get("summary", {})
    metadata = data.get("metadata", {})

    # Reconstruct EvalResult objects
    from llm_eval.models import EvalResult, MetricResult
    results: list[EvalResult] = []
    for r in results_data:
        metrics = [
            MetricResult(name=m["name"], score=m["score"], details=m.get("details", {}))
            for m in r.get("metrics", [])
        ]
        results.append(EvalResult(sample_index=r["sample_index"], metrics=metrics))

    # Auto-detect format from output extension
    if to_format is None:
        if output_path:
            ext = os.path.splitext(output_path)[1].lower()
            to_format = _FORMAT_EXTENSIONS.get(ext, "terminal")
        else:
            to_format = "terminal"

    content = _format_report(to_format, results, summary, metadata)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        click.echo(f"✅ Exported to {output_path} ({to_format})")
    else:
        click.echo(content)


# ─── Config subgroup ─────────────────────────────────────────────────────


@main.group("config", invoke_without_command=True)
@click.pass_context
def config_group(ctx: click.Context) -> None:
    """Configuration utilities (schema generation, etc.)."""
    if ctx.invoked_subcommand is None:
        click.echo("Use: llm-eval config schema")


@config_group.command("schema")
@click.option("--output", "-o", "output_path", default=None, help="Write schema to file.")
def config_schema(output_path: str | None) -> None:
    """Export JSON Schema for the evaluation config YAML.

    Useful for editor autocompletion and validation.
    """
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "llm-eval Configuration",
        "description": "Configuration schema for llm-eval evaluation tool.",
        "type": "object",
        "properties": {
            "extends": {
                "type": "string",
                "description": "Path to a base config file to inherit from.",
            },
            "judge": {
                "type": "object",
                "description": "LLM judge model configuration.",
                "properties": {
                    "model": {
                        "type": "string",
                        "default": "gpt-4o",
                        "description": "Judge model identifier.",
                    },
                    "base_url": {
                        "type": ["string", "null"],
                        "description": "Custom API endpoint URL.",
                    },
                    "api_key": {
                        "type": ["string", "null"],
                        "description": "Explicit API key (falls back to env vars).",
                    },
                    "temperature": {
                        "type": "number",
                        "default": 0.0,
                        "minimum": 0,
                        "maximum": 2,
                        "description": "Sampling temperature.",
                    },
                    "max_retries": {
                        "type": "integer",
                        "default": 3,
                        "minimum": 1,
                        "description": "Maximum retry attempts on failure.",
                    },
                    "timeout": {
                        "type": "integer",
                        "default": 60,
                        "minimum": 1,
                        "description": "Request timeout in seconds.",
                    },
                },
                "required": ["model"],
            },
            "defaults": {
                "type": "object",
                "description": "Default evaluation settings.",
                "properties": {
                    "threshold": {
                        "type": "number",
                        "default": 0.7,
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Pass/fail score threshold.",
                    },
                    "parallel": {
                        "type": "integer",
                        "default": 1,
                        "minimum": 1,
                        "description": "Number of parallel evaluations.",
                    },
                    "output_format": {
                        "type": "string",
                        "default": "terminal",
                        "enum": ["terminal", "json", "csv", "html", "markdown", "junit"],
                        "description": "Report output format.",
                    },
                    "metric_weights": {
                        "type": "object",
                        "description": "Per-metric weights for overall score.",
                        "additionalProperties": {"type": "number"},
                    },
                },
            },
            "evaluations": {
                "type": "array",
                "description": "List of evaluation definitions.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Evaluation run name.",
                        },
                        "dataset": {
                            "type": "string",
                            "description": "Path to JSONL or CSV dataset file.",
                        },
                        "metrics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of metric names to evaluate.",
                        },
                        "threshold": {
                            "type": ["number", "null"],
                            "minimum": 0,
                            "maximum": 1,
                            "description": "Per-evaluation threshold override.",
                        },
                        "parallel": {
                            "type": ["integer", "null"],
                            "minimum": 1,
                            "description": "Per-evaluation parallelism override.",
                        },
                        "metric_weights": {
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                        },
                        "metric_options": {
                            "type": "object",
                            "description": "Per-metric custom options.",
                        },
                    },
                    "required": ["name", "dataset", "metrics"],
                },
            },
            "custom_metrics": {
                "type": "array",
                "description": "Custom metric plugin definitions.",
                "items": {
                    "type": "object",
                    "properties": {
                        "module": {"type": "string"},
                        "class": {"type": "string"},
                    },
                    "required": ["module", "class"],
                },
            },
        },
        "required": ["evaluations"],
    }

    content = json.dumps(schema, indent=2, ensure_ascii=False)
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        click.echo(f"✅ Schema written to: {output_path}")
    else:
        click.echo(content)
