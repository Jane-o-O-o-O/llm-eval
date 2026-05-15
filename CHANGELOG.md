# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-15

### Added
- **`llm-eval cache stats|clear|purge` commands**: Full cache management from the CLI. View entry count and size (`stats`), clear all entries (`clear --yes`), or remove entries older than N days (`purge --older-than 30 --yes`).
- **`llm-eval export` command**: Convert JSON evaluation reports to other formats (HTML, CSV, Markdown, JUnit XML, terminal). Supports `--to` format selection and auto-detection from output file extension.
- **`--filter` option on `run`**: Filter dataset samples by metadata field values (e.g. `--filter metadata.category=tech`). Works in both dry-run and live evaluation modes.
- **`--timeout` option on `run`**: Override judge timeout per evaluation run without editing config files.
- **`llm-eval config schema` command**: Export JSON Schema for the evaluation config YAML format. Useful for editor autocompletion and validation. Supports `--output` to write to file.
- **Sync SDK wrappers**: `evaluate_sync()` and `evaluate_file_sync()` convenience functions for non-async contexts. Internally wrap the async versions with `asyncio.run()`.
- 520 tests (up from 491).

### Changed
- `__init__.py` now exports `evaluate_sync` and `evaluate_file_sync` from the top-level package.
- CLI help now shows `cache`, `export`, and `config` commands.

## [0.9.0] - 2026-05-15

### Added
- **Judge cache integration**: Judge responses are now automatically cached via SQLite (`~/.llm-eval/cache.db`). The `Judge` class accepts `cache` and `use_cache` parameters. The `Evaluator` automatically wires up the cache for all metrics.
- **API key authentication**: Judge now sends `Authorization: Bearer` headers. Supports explicit `api_key` in config, or falls back to environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`). Provider auto-detection based on `base_url`.
- **`llm-eval doctor` command**: Environment diagnostics tool that checks Python version, dependencies, API keys (with masked display), available metrics, and cache statistics.
- **Streaming dataset loader**: `stream_jsonl()` function for memory-efficient lazy iteration over large JSONL files.
- **`count_samples()` function**: Count samples in JSONL/CSV files without loading them into memory.
- **Metric options support**: Metrics accept `metric_options` dict for custom configuration (e.g., custom prompt templates). Config YAML supports `metric_options` per evaluation.
- **PEP 561 compliance**: Added `py.typed` marker file for type checker support.
- **`api_key` in JudgeConfig**: Explicit API key field in configuration with environment variable fallback.
- 491 tests (up from 431).

### Changed
- `Judge.__init__()` now accepts `cache` and `use_cache` parameters.
- `Metric.__init__()` now accepts `cache`, `use_cache`, and `metric_options` parameters.
- `Evaluator.__init__()` now accepts `use_cache` and `metric_options` parameters.
- `JudgeConfig` now includes `api_key: str | None` field.
- `dataset.py` exports `stream_jsonl` and `count_samples` in `__all__`.

## [0.8.0] - 2026-05-14

### Added
- **JUnit XML report format**: `--output junit` generates JUnit XML reports for CI/CD integration (GitHub Actions, Jenkins, GitLab CI). Each metric becomes a `<testsuite>`, each sample a `<testcase>`. Samples below threshold produce `<failure>` elements.
- **`--model` / `-m` CLI option**: Override the judge model directly from the command line without editing config files (e.g. `llm-eval run -c config.yaml -m claude-3-opus`).
- **Score distribution statistics**: Summary now includes median, 25th/75th percentiles (p25/p75), standard deviation, min and max scores for richer quality insights.
- **`llm-eval history trend` command**: Visualize score trends across evaluation runs with an ASCII sparkline chart and chronological run table.
- **Config file inheritance (`extends`)**: Configs can now extend base configs with `extends: base.yaml`. Child values override base values. Dicts are deep-merged recursively.
- **`llm-eval dataset convert` command**: Convert datasets between JSONL and CSV formats. Format is auto-detected from file extension.
- **`llm-eval metrics create` command**: Scaffold custom metric template files with a Metric subclass skeleton, ready to fill in with custom evaluation logic.
- 431 tests (up from 401).

## [0.7.0] - 2026-05-14

### Added
- **Python SDK**: `from llm_eval import evaluate, evaluate_file` — programmatic API for evaluating samples without the CLI. Returns `EvalOutput` with results, summary, and pre-formatted reports.
- **Judge response cache**: SQLite-backed cache (`~/.llm-eval/cache.db`) that stores LLM judge responses. Avoids redundant API calls during iterative development. Configurable max entries with LRU eviction.
- **`llm-eval dataset` subcommand**: Three sub-commands — `info` (dataset statistics), `validate` (check for common issues), `sample` (show random samples). Works with JSONL and CSV datasets.
- **Markdown report format**: `--output markdown` generates GitHub-flavored Markdown reports suitable for PR comments and documentation. Auto-omits per-sample tables for large datasets.
- **Evaluation run history**: All `llm-eval run` results are automatically saved to `~/.llm-eval/history/`. Browse with `llm-eval history`.
- **`--tag` option**: Tag evaluation runs for easy organization (e.g. `--tag baseline`, `--tag experiment-1`).
- **`--no-cache` option**: Disable judge response caching for a specific run.
- **`--save-history` / `--no-save-history` option**: Control whether runs are saved to history.
- **`llm-eval history` command**: Browse past evaluation runs with filtering by tag and limit.
- 401 tests (up from 319).

### Changed
- `__init__.py` now exports `evaluate`, `evaluate_file`, and `EvalOutput` for SDK usage.
- `cli.py` valid output formats now include `markdown`.
- Version bumped to 0.7.0.

## [0.6.0] - 2026-05-13

### Added
- **CSV dataset support**: `load_csv()` loader with pipe-separated and JSON array context formats. Auto-detection via `load_dataset()` based on file extension.
- **Report metadata**: All report formats (terminal, JSON, CSV, HTML) now include metadata — timestamp, version, Python version, platform, config path, git hash.
- **Multi-format output**: `--output json,html` comma-separated format support. Generates multiple report files in one run.
- **Config presets**: `llm-eval init --preset rag|chatbot|summarization` for quick project scaffolding with curated metric configurations.
- **`llm-eval presets` command**: List available presets with descriptions.
- **`--verbose` / `-v` flag for `metrics` command**: Shows detailed metric descriptions.
- **HTML comparison report**: `llm-eval compare` now supports `--output html` with side-by-side SVG bar charts and delta indicators.
- **Expandable HTML details**: HTML reports now show per-sample metric reasoning/details in collapsible sections with 🔍 toggle buttons.
- 319 tests (up from 263).

### Changed
- `dataset.py` now exports `load_csv()` and `load_dataset()` in addition to `load_jsonl()`.
- Report functions accept optional `metadata` parameter.
- `compare.py` gained `format_html_comparison()` function.
- Version bumped to 0.6.0.

## [0.5.0] - 2026-05-13

### Added
- **`--quiet` / `-q` flag**: Minimal output mode for CI/CD pipelines. Suppresses verbose progress and report output, only shows pass/fail result.
- **SVG score distribution chart**: HTML reports now include an inline SVG histogram showing the distribution of per-sample scores across score buckets.
- **`py.typed` marker**: PEP 561 compliance for type checker support (mypy, pyright).
- **GitHub Actions CI workflow**: Automated testing on Python 3.10/3.11/3.12 with ruff linting.
- **Ruff lint rules**: Added select rules (E, F, W, I, N, UP, B, SIM) and format configuration in `pyproject.toml`.
- **Version consistency test**: Validates `__version__` matches `pyproject.toml` version.
- **`__main__` entry point tests**: Tests for `python -m llm_eval` commands.
- 263 tests (up from 252).

### Changed
- Version bumped to 0.5.0.

## [0.4.0] - 2026-05-13

### Added
- **hallucination metric**: Detects fabricated claims in answers not supported by retrieved context. Reports individual hallucinated claims in details.
- **answer_similarity metric**: LLM-judge semantic similarity between generated answer and reference. Requires a reference answer in the sample.
- **`__main__.py`**: Support running as `python -m llm_eval`.
- 10 built-in metrics total (up from 8).

### Fixed
- **Test suite hanging**: CLI and regression tests that invoked `run` without `--dry-run` made real HTTP calls and hung indefinitely. Now uses `unittest.mock.patch` to mock `httpx.AsyncClient` for fast test execution.
- All 252 tests now complete in under 1 second (previously 2 test files would timeout).

## [0.3.0] - 2026-05-12

### Added
- **coherence metric**: Evaluates answer quality — structure, fluency, and logical flow. Useful for catching answers that are factually correct but poorly written.
- **`--sample N` CLI flag**: Randomly sample N items from dataset for quick evaluation during development. Supports `--seed` for reproducibility.
- **`conftest.py`**: Shared test fixtures (`sample`, `sample_with_reference`, `multiple_samples`).
- **`_judge_call` in base `Metric` class**: All LLM-based metrics now inherit a shared `_judge_call()` method. Eliminates code duplication across 6 metrics.
- **`judge_config` propagation**: `JudgeConfig` from YAML config is now passed through `Evaluator` → `get_default_registry()` → each metric instance. Custom base_url, model, timeout settings are actually used.

### Fixed
- **Version sync**: `__init__.py` version now matches `pyproject.toml` (was 0.1.0 vs 0.2.0).
- **Async ABC**: `Metric.evaluate` is now properly declared as `async def` in the abstract base class, matching all implementations.

### Changed
- `Metric.__init__` now accepts optional `judge_config: JudgeConfig` parameter.
- `get_default_registry()` now accepts optional `judge_config` parameter to pass to all metrics.
- `Evaluator.__init__` now accepts optional `judge_config` parameter.
- `FormatComplianceMetric.__init__` uses `**kwargs` pattern to forward judge_config to base.
- `AnswerCorrectnessMetric.__init__` uses `**kwargs` pattern to forward judge_config to base.
- Version bumped to 0.3.0.
- 227 tests (up from 213).

## [0.2.0] - 2026-05-12

### Added
- **answer_correctness metric**: Hybrid token-overlap + LLM-judge evaluation. Uses weighted blend (40% token F1, 60% judge) when reference is available, falls back to judge-only otherwise.
- **Metric weights**: Configure per-metric weights in YAML config (`defaults.metric_weights`) to influence overall score calculation. Metrics without explicit weights default to 1.0.
- **Dry-run mode**: `llm-eval run --dry-run` validates config, checks metrics/datasets, and prints the execution plan without running evaluations.
- **Custom metric plugins**: Load external metrics via `custom_metrics` in YAML config. Specify Python module path and class name for dynamic loading.
- **CHANGELOG.md**: This file, tracking all notable changes.
- **GitHub Actions CI**: Automated testing on Python 3.10, 3.11, 3.12 with linting via ruff.

### Fixed
- **compare.py**: Removed `_path` attribute injection hack. Report paths are now passed as explicit `path_a`/`path_b` parameters to `compare_reports()`.

### Changed
- `Evaluator.__init__` now accepts optional `metric_weights` parameter.
- `EvalConfig` gained `metric_weights` and `custom_metrics` fields.

## [0.1.0] - 2026-05-11

### Added
- Initial release of llm-eval.
- Core evaluation engine with async support and parallel execution.
- 6 built-in metrics: faithfulness, answer_relevancy, context_precision, context_recall, format_compliance, toxicity.
- LLM-as-Judge adapter (OpenAI-compatible API with retry logic).
- YAML config file support.
- CLI commands: `init`, `run`, `metrics`, `validate`, `compare`.
- Report formats: terminal, JSON, CSV, HTML.
- Regression detection mode (`--fail-on regression`).
- Progress bar support via tqdm.
- 176 unit tests.
