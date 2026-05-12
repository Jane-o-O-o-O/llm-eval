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
| 📊 **8 Built-in Metrics** | Faithfulness, answer relevancy, correctness, coherence, context precision/recall, format compliance, toxicity |
| 📄 **Rich Reports** | JSON, CSV, HTML, and terminal summary output |
| ⚡ **Parallel Evaluation** | Concurrent sample evaluation with progress bars |
| 📉 **Regression Detection** | Compare against baselines to catch quality drops |
| 🔌 **Pluggable** | Custom metrics, CI/CD integration |
| 🪶 **Lightweight** | Minimal dependencies — `click` + `httpx` + your API key |

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
coherence              Answer quality: structure, fluency, and logical flow
context_precision      Signal-to-noise ratio in retrieved context
context_recall         Coverage of reference by retrieved context
faithfulness           Factual consistency between answer and context
format_compliance      Output matches required schema/format (deterministic)
toxicity               Detects harmful, biased, or offensive content

Total: 8 metrics
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
| **coherence** | Answer quality: structure, fluency, and logical flow | LLM Judge |
| **context_precision** | Signal-to-noise ratio in retrieved context | LLM Judge |
| **context_recall** | Coverage of reference answer by retrieved context | LLM Judge |
| **format_compliance** | Output matches required schema/format | Deterministic |
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
