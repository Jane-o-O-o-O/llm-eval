# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
