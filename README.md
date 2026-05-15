<div align="center">

# 🧪 llm-eval

**Lightweight CLI for evaluating LLM applications**

[![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Evaluate RAG pipelines and LLM outputs — from the command line.*

[Getting Started](#-quick-start) · [Metrics](#-metrics) · [Commands](#-commands) · [Configuration](#%EF%B8%8F-configuration)

</div>

---

## ✨ Why llm-eval?

Building LLM-powered apps is easy. **Knowing if they work well is hard.**

`llm-eval` gives you a fast, repeatable way to measure quality using **LLM-as-Judge** — no heavyweight frameworks, no complex infrastructure. Just a CLI and a config file.

| Feature | Description |
|---|---|
| 🚀 **Fast Setup** | Install in seconds, evaluate in minutes |
| 🎯 **LLM-as-Judge** | Use GPT-4, Claude, Gemini, or any OpenAI-compatible model as evaluator |
| 📊 **10 Built-in Metrics** | Faithfulness, answer relevancy, correctness, coherence, context precision/recall, format compliance, toxicity, hallucination, answer similarity |
| 📄 **Rich Reports** | JSON, CSV, HTML, JUnit XML, and terminal summary output |
| ⚡ **Parallel Evaluation** | Concurrent sample evaluation with progress bars |
| 📉 **Regression Detection** | Compare against baselines to catch quality drops |
| 🔌 **Pluggable** | Custom metrics, CI/CD integration |
| 🤫 **Quiet Mode** | `--quiet` flag for CI/CD pipelines |
| 📁 **CSV Support** | Load datasets from CSV or JSONL, auto-detected |
| 🎯 **Config Presets** | `llm-eval init --preset rag|chatbot|summarization` |
| 📋 **Report Metadata** | Timestamps, versions, git hash embedded in all reports |
| 🔀 **Multi-Format Output** | `--output json,html` for multiple report files at once |
| 🪶 **Lightweight** | Minimal dependencies — `click` + `httpx` + your API key |
| 🐍 **Python SDK** | `from llm_eval import evaluate` for programmatic use |
| 💾 **Judge Cache** | SQLite-backed cache saves API costs during development |
| 📂 **Dataset Tools** | `llm-eval dataset info|validate|sample` to inspect datasets |
| 📝 **Markdown Reports** | `--output markdown` for GitHub PR comments |
| 🏷️ **Run History** | Automatic run tracking with `--tag` and `llm-eval history` |
| 🔧 **Config Inheritance** | `extends: base.yaml` for DRY configuration |
| 📊 **Score Distribution** | Median, p25/p75, std dev in evaluation summaries |
| 📈 **History Trends** | `llm-eval history trend` with ASCII sparkline charts |
| 💾 **Cache Management** | `llm-eval cache stats\|clear\|purge` to manage judge cache |
| 📤 **Report Export** | `llm-eval export` to convert reports between formats |
| 🔍 **Sample Filtering** | `--filter metadata.category=tech` to evaluate subsets |
| ⏱️ **Timeout Override** | `--timeout 120` to override judge timeout per run |
| 📋 **Config Schema** | `llm-eval config schema` for editor autocompletion |
| 🔄 **Sync SDK** | `evaluate_sync()` for non-async Python contexts |
| 📊 **Per-Metric Stats** | Median, p25/p75, std dev for every metric |
| ⚙️ **CLI Overrides** | `--set judge.model=claude-3-opus` without editing YAML |
| 🔍 **Metric-Aware Validation** | `dataset validate --metrics` checks field requirements |
| 📊 **History Diff** | `history diff` to compare two specific runs |

---

## 📦 Installation

```bash
# from source
git clone https://github.com/Jane-o-O-o-O/llm-eval.git
cd llm-eval
pip install -e .
```

Set your judge model API key:

```bash
export OPENAI_API_KEY="sk-..."        # for OpenAI (default judge)
```

---

## 🚀 Quick Start

### 1. Initialize a project

```bash
llm-eval init --output evals.yaml
# Or use a preset for common scenarios:
llm-eval init --preset rag --output evals.yaml
llm-eval init --preset chatbot --output evals.yaml
llm-eval init --preset summarization --output evals.yaml
```

This generates a starter config and sample dataset:

```yaml
# evals.yaml
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
```

### 2. Add test samples

Each line in your `.jsonl` dataset needs at minimum `query`, `context`, and `answer`:

```jsonl
{"query": "What is the refund policy?", "context": ["Refunds are processed within 5 business days."], "answer": "Refunds take up to 5 business days.", "reference": "Refunds are processed within 5 business days."}
{"query": "How do I reset my password?", "context": ["Click 'Forgot Password' on the login page."], "answer": "Go to the login page and click 'Forgot Password'.", "reference": "Navigate to login and click 'Forgot Password' to receive a reset email."}
```

### 3. Run evaluation

```bash
llm-eval run --config evals.yaml
```

```
┌─────────────────────────────────────────────────┐
│          🧪 llm-eval — Evaluation Report         │
├─────────────────────┬───────────┬───────────────┤
│ Metric              │ Score     │ Status        │
├─────────────────────┼───────────┼───────────────┤
│ faithfulness        │ 0.95      │ ✅ PASS       │
│ answer_relevancy    │ 0.88      │ ✅ PASS       │
├─────────────────────┼───────────┼───────────────┤
│ Overall             │ 0.92      │ ✅ PASS       │
└─────────────────────┴───────────┴───────────────┘
Evaluated 2 samples (2 passed, 0 failed)
```

---

## 📖 Commands

### `llm-eval init`
Initialize a new evaluation project with sample config and dataset.

```bash
llm-eval init --output evals.yaml
```

### `llm-eval run`
Run evaluations based on a config file.

```bash
# Basic run
llm-eval run --config evals.yaml

# Output as JSON to file
llm-eval run --config evals.yaml --output json --report results.json

# Override threshold
llm-eval run --config evals.yaml --threshold 0.85

# Parallel evaluation (5 concurrent)
llm-eval run --config evals.yaml --parallel 5

# Regression check against a baseline
llm-eval run --config evals.yaml --fail-on regression --baseline previous_results.json --tolerance 0.05

# Quick evaluation on a random subset (for fast iteration)
llm-eval run --config evals.yaml --sample 10

# Reproducible sample with seed
llm-eval run --config evals.yaml --sample 10 --seed 42

# Quiet mode for CI/CD (minimal output, exit code indicates pass/fail)
llm-eval run --config evals.yaml --quiet
```

### `llm-eval metrics`
List all available evaluation metrics.

```bash
llm-eval metrics
```

```
📋 Available Metrics

Name                   Description
────────────────────────────────────────────────────────────
answer_relevancy       How well the answer addresses the query
answer_correctness     Hybrid token-overlap + LLM-judge correctness against reference
answer_similarity      Semantic similarity between answer and reference answer
coherence              Answer quality: structure, fluency, and logical flow
context_precision      Signal-to-noise ratio in retrieved context
context_recall         Coverage of reference by retrieved context
faithfulness           Factual consistency between answer and context
format_compliance      Output matches required schema/format (deterministic)
hallucination          Degree of fabricated claims not supported by context
toxicity               Detects harmful, biased, or offensive content

Total: 10 metrics
```

### `llm-eval validate`
Validate an evaluation config file.

```bash
llm-eval validate evals.yaml
```

Checks: valid YAML structure, known metrics, existing datasets, valid thresholds.

### `llm-eval compare`
Compare two evaluation reports side by side.

```bash
llm-eval compare baseline.json current.json --label-a "v1" --label-b "v2"

# JSON output
llm-eval compare baseline.json current.json --output json

# Save to file
llm-eval compare baseline.json current.json --report comparison.html
```

---

## 📊 Metrics

### RAG & Quality Metrics

| Metric | Description | Method |
|---|---|---|
| **faithfulness** | Factual consistency between answer and retrieved context | LLM Judge |
| **answer_relevancy** | How well the answer addresses the original query | LLM Judge |
| **answer_correctness** | Hybrid token-overlap + LLM-judge correctness against reference | Token + LLM Judge |
| **answer_similarity** | Semantic similarity between answer and reference answer | LLM Judge |
| **coherence** | Answer quality: structure, fluency, and logical flow | LLM Judge |
| **context_precision** | Signal-to-noise ratio in retrieved context | LLM Judge |
| **context_recall** | Coverage of reference answer by retrieved context | LLM Judge |
| **format_compliance** | Output matches required schema/format | Deterministic |
| **hallucination** | Degree of fabricated claims not supported by context | LLM Judge |
| **toxicity** | Detects harmful, biased, or offensive content | Pattern + LLM Judge |

### Custom Metrics

Define your own metrics in Python by extending the `Metric` base class:

```python
from llm_eval.metrics import Metric, MetricResult
from llm_eval.models import Sample

class CostEfficiency(Metric):
    name = "cost_efficiency"
    description = "Quality-adjusted cost per query"

    async def evaluate(self, sample: Sample) -> MetricResult:
        cost = sample.metadata.get("total_cost", 0)
        score = 1.0 / max(cost, 0.001)
        return MetricResult(name=self.name, score=min(score, 1.0), details={"cost": cost})
```

---

## ⚙️ Configuration

Full configuration reference for `evals.yaml`:

```yaml
judge:
  model: gpt-4o              # any OpenAI-compatible model
  base_url: null              # custom endpoint (e.g., Ollama, vLLM)
  temperature: 0
  max_retries: 3
  timeout: 60

defaults:
  threshold: 0.7              # default pass/fail threshold
  parallel: 5                 # concurrent evaluations
  output_format: terminal     # terminal | json | csv | html

evaluations:
  - name: "My Pipeline"
    dataset: data.jsonl
    metrics: [faithfulness, answer_relevancy, toxicity]
    threshold: 0.8            # override default
    parallel: 3               # override default parallelism
```

---

## 🗂️ Dataset Format

```jsonl
{
  "query": "user question",
  "context": ["retrieved chunk 1", "retrieved chunk 2"],
  "answer": "generated answer to evaluate",
  "reference": "ground truth answer (optional)",
  "metadata": {
    "expected_format": "json",     // optional: for format_compliance
    "min_words": 10,               // optional: word count bounds
    "max_words": 200
  }
}
```

**Required fields:** `query`, `context`, `answer`
**Optional fields:** `reference`, `metadata`

### CSV Format

You can also use CSV files. Context supports pipe-separated or JSON array formats:

```csv
query,context,answer,reference
What is AI?,"AI is a field | It studies intelligence",AI is a branch of CS,Artificial Intelligence is...
What is ML?,Machine learning is a subset,ML learns from data,
```

Format is auto-detected by file extension (`.csv` vs `.jsonl`/`.ndjson`).

---

## 📈 Output Formats

```bash
# Terminal (default) — colored table summary
llm-eval run --config evals.yaml

# JSON — machine-readable, good for CI/CD
llm-eval run --config evals.yaml --output json --report results.json

# CSV — spreadsheet-friendly
llm-eval run --config evals.yaml --output csv --report results.csv

# HTML — shareable report with dark theme
llm-eval run --config evals.yaml --output html --report report.html

# Multiple formats at once — generates report.json and report.html
llm-eval run --config evals.yaml --output json,html --report report

# Markdown — for GitHub PR comments
llm-eval run --config evals.yaml --output markdown --report report.md
```

---

## 🎯 Config Presets

Quick start with curated metric configurations for common use cases:

```bash
# RAG pipeline (faithfulness + relevancy + context metrics)
llm-eval init --preset rag --output evals.yaml

# Chatbot quality (coherence + relevancy + toxicity)
llm-eval init --preset chatbot --output evals.yaml

# Summarization (faithfulness + hallucination + similarity)
llm-eval init --preset summarization --output evals.yaml

# List all available presets
llm-eval presets
```

---

## 🐍 Python SDK

Use llm-eval programmatically without the CLI:

```python
import asyncio
from llm_eval import evaluate

async def main():
    output = await evaluate(
        samples=[
            {"query": "What is Python?", "context": ["Python is a language."], "answer": "A programming language."},
        ],
        metrics=["faithfulness", "answer_relevancy"],
        model="gpt-4o",
    )
    print(f"Score: {output.overall_score:.2f}")
    print(f"Passed: {output.passed}")
    print(output.terminal)  # pre-formatted report

asyncio.run(main())
```

Or evaluate from a file:

```python
from llm_eval import evaluate_file

output = await evaluate_file(
    path="samples.jsonl",
    config="evals.yaml",
)
```

---

## 📂 Dataset Tools

Inspect and validate your evaluation datasets:

```bash
# Show dataset statistics
llm-eval dataset info samples.jsonl

# Validate for common issues
llm-eval dataset validate samples.jsonl

# Preview random samples
llm-eval dataset sample samples.jsonl -n 5
```

---

## 🏷️ Run History

Track and compare evaluation runs over time:

```bash
# Tag a baseline run
llm-eval run --config evals.yaml --tag baseline

# Tag an experiment
llm-eval run --config evals.yaml --tag experiment-1

# Browse history
llm-eval history
llm-eval history --tag baseline
llm-eval history -n 5 --details
```

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   CLI (click) │────▶│  Evaluator   │────▶│  Judge (LLM) │
│              │     │   Engine     │     │   Adapter    │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                    ┌───────▼───────┐
                    │    Metrics    │
                    │   Framework   │
                    └───────┬───────┘
                            │
                    ┌───────▼───────┐
                    │   Report      │
                    │   Generator   │
                    └───────────────┘
```

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📜 License

[MIT](LICENSE) — use it freely in personal and commercial projects.

---

<div align="center">

⭐ Star this repo if you find it useful!

</div>
