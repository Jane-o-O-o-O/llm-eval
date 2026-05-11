# Contributing to llm-eval

Thank you for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/Jane-o-O-o-O/llm-eval.git
cd llm-eval

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .

# Install dev dependencies
pip install pytest pytest-asyncio
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_toxicity.py -v

# Run with coverage (if installed)
python -m pytest tests/ --cov=llm_eval
```

## Project Structure

```
llm_eval/
├── __init__.py          # Package entry, version, public exports
├── cli.py               # Click CLI commands (init, run, metrics, validate, compare)
├── models.py            # Data models (Sample, MetricResult, EvalResult, etc.)
├── evaluator.py         # Core evaluation engine with parallel support
├── dataset.py           # JSONL dataset loader
├── report.py            # Report formatters (terminal, JSON, CSV, HTML)
├── regression.py        # Baseline comparison for regression detection
├── compare.py           # Side-by-side report comparison
├── judge/
│   └── __init__.py      # LLM judge adapter (OpenAI-compatible API)
└── metrics/
    ├── __init__.py       # Metric base class, registry, default registration
    ├── faithfulness.py   # Faithfulness metric
    ├── answer_relevancy.py
    ├── context_precision.py
    ├── context_recall.py
    ├── format_compliance.py  # Deterministic format checker
    └── toxicity.py       # Toxicity detection (pattern + LLM judge)

tests/
├── test_cli.py
├── test_models.py
├── test_evaluator.py
├── test_dataset.py
├── test_metrics_base.py
├── test_faithfulness.py
├── test_answer_relevancy.py
├── test_context_precision.py
├── test_context_recall.py
├── test_format_compliance.py
├── test_toxicity.py
├── test_report.py
├── test_html_report.py
├── test_regression.py
├── test_compare.py
└── test_judge.py
```

## Adding a New Metric

1. Create `llm_eval/metrics/your_metric.py`:

```python
"""Your metric description."""

from __future__ import annotations

from llm_eval.metrics import Metric, MetricResult
from llm_eval.models import Sample


class YourMetric(Metric):
    name = "your_metric"
    description = "Short description of what this metric measures"

    async def evaluate(self, sample: Sample) -> MetricResult:
        # Your evaluation logic here
        score = 1.0  # compute score between 0.0 and 1.0
        return MetricResult(
            name=self.name,
            score=score,
            details={"reasoning": "explanation"},
        )
```

2. Register it in `llm_eval/metrics/__init__.py`:
   - Import your metric class in `get_default_registry()`
   - Call `registry.register(YourMetric())`

3. Add tests in `tests/test_your_metric.py`

4. Run tests: `python -m pytest tests/ -v`

## Code Style

- Use type annotations everywhere (`from __future__ import annotations`)
- Write docstrings for all public classes and methods
- Keep functions short and focused
- Use async/await for anything that calls external services
- Tests should be deterministic — mock LLM calls

## Commit Messages

Use Chinese commit messages in this format:

```
类型: 简要描述

- feat: 新功能
- fix: 修复
- docs: 文档
- test: 测试
- refactor: 重构
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/your-feature`)
3. Make your changes
4. Run tests (`python -m pytest tests/ -v`)
5. Commit with a descriptive message
6. Push and open a Pull Request
