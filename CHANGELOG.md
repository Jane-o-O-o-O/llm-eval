# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
