<div align="center">

# 🧪 llm-eval

**Lightweight CLI for evaluating LLM applications**

[![PyPI version](https://img.shields.io/pypi/v/llm-eval?color=blue&logo=pypi&logoColor=white)](https://pypi.org/project/llm-eval/)
[![Python](https://img.shields.io/pypi/pyversions/llm-eval?logo=python&logoColor=white)](https://pypi.org/project/llm-eval/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/your-org/llm-eval/ci.yml?branch=main&logo=github)](https://github.com/your-org/llm-eval/actions)
[![codecov](https://img.shields.io/codecov/c/github/your-org/llm-eval?logo=codecov)](https://codecov.io/gh/your-org/llm-eval)
[![Docs](https://img.shields.io/badge/docs-readthedocs-blue?logo=readthedocs)](https://llm-eval.readthedocs.io)

*Evaluate RAG pipelines, AI agents, and prompt templates — from the command line.*

[Getting Started](#-quick-start) · [Metrics](#-metrics) · [Examples](#-usage-examples) · [Contributing](#-contributing)

</div>

---

## ✨ Why llm-eval?

Building LLM-powered apps is easy. **Knowing if they work well is hard.**

`llm-eval` gives you a fast, repeatable way to measure quality using **LLM-as-Judge** — no heavyweight frameworks, no complex infrastructure. Just a CLI and a config file.

| Feature | Description |
|---|---|
| 🚀 **Fast Setup** | Install in seconds, evaluate in minutes |
| 🎯 **LLM-as-Judge** | Use GPT-4, Claude, Gemini, or any OpenAI-compatible model as evaluator |
| 📦 **3 Evaluation Modes** | RAG quality, Agent accuracy, Prompt effectiveness |
| 📊 **Rich Reports** | JSON, CSV, HTML, and terminal summary output |
| 🔌 **Pluggable** | Custom metrics, custom judges, CI/CD integration |
| 🪶 **Lightweight** | Minimal dependencies — `click` + `httpx` + your API key |

---

## 📦 Installation

```bash
# pip
pip install llm-eval

# pipx (recommended for CLI-only use)
pipx install llm-eval

# from source
git clone https://github.com/your-org/llm-eval.git
cd llm-eval
pip install -e .
```

Set your judge model API key:

```bash
export OPENAI_API_KEY="sk-..."        # for OpenAI (default judge)
export ANTHROPIC_API_KEY="sk-ant-..."  # for Claude judge
```

---

## 🚀 Quick Start

### 1. Create a dataset

```bash
llm-eval init --output evals.yaml
```

This generates a starter config:

```yaml
# evals.yaml
judge:
  model: gpt-4o
  temperature: 0

evaluations:
  - name: "My RAG Pipeline"
    type: rag
    dataset: samples.jsonl
    metrics:
      - faithfulness
      - answer_relevancy
      - context_precision

  - name: "Customer Support Agent"
    type: agent
    dataset: agent_tasks.jsonl
    metrics:
      - task_completion
      - tool_accuracy

  - name: "Summary Prompts v2"
    type: prompt
    dataset: prompt_tests.jsonl
    metrics:
      - correctness
      - conciseness
      - tone
```

### 2. Add test samples

```jsonl
{"query": "What is the refund policy?", "context": ["Refunds are processed within 5 business days..."], "answer": "Refunds take up to 5 business days.", "reference": "Refunds are processed within 5 business days."}
{"query": "How do I reset my password?", "context": ["Click 'Forgot Password' on the login page..."], "answer": "Go to the login page and click 'Forgot Password'.", "reference": "Navigate to login and click 'Forgot Password' to receive a reset email."}
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
│ context_precision   │ 0.72      │ ⚠️  WARN       │
├─────────────────────┼───────────┼───────────────┤
│ Overall             │ 0.85      │ ✅ PASS       │
└─────────────────────┴───────────┴───────────────┘
Evaluated 42 samples in 34.2s using gpt-4o
```

---

## 📖 Usage Examples

### Evaluate a RAG Pipeline

```bash
llm-eval run \
  --type rag \
  --dataset qa_pairs.jsonl \
  --judge gpt-4o \
  --metrics faithfulness,answer_relevancy,context_recall \
  --output report.json
```

### Evaluate an Agent

```bash
llm-eval run \
  --type agent \
  --dataset agent_tasks.jsonl \
  --metrics task_completion,tool_accuracy,reasoning \
  --judge claude-3-opus
```

### Compare Prompt Variants

```bash
llm-eval compare \
  --config experiments.yaml \
  --variants "v1:prompt_v1.yaml,v2:prompt_v2.yaml,v3:prompt_v3.yaml" \
  --metrics correctness,conciseness \
  --output comparison.html
```

### CI/CD Integration

```bash
# Fail pipeline if score drops below threshold
llm-eval run \
  --config evals.yaml \
  --threshold 0.85 \
  --output json \
  --fail-on regression
```

```yaml
# GitHub Actions example
- name: Evaluate LLM quality
  run: |
    pip install llm-eval
    llm-eval run --config evals.yaml --threshold 0.80 --fail-on regression
```

---

## 📊 Metrics

### RAG Metrics

| Metric | Description | Score Range | Method |
|---|---|---|---|
| **faithfulness** | Factual consistency between answer and retrieved context | 0.0 – 1.0 | LLM Judge |
| **answer_relevancy** | How well the answer addresses the original query | 0.0 – 1.0 | LLM Judge + Embedding |
| **context_precision** | Signal-to-noise ratio in retrieved context | 0.0 – 1.0 | LLM Judge |
| **context_recall** | Coverage of reference answer by retrieved context | 0.0 – 1.0 | LLM Judge |
| **answer_correctness** | Factual alignment with ground truth reference | 0.0 – 1.0 | LLM Judge + Token Overlap |

### Agent Metrics

| Metric | Description | Score Range | Method |
|---|---|---|---|
| **task_completion** | Whether the agent achieved the stated goal | 0.0 – 1.0 | LLM Judge |
| **tool_accuracy** | Correctness of tool/function selections | 0.0 – 1.0 | LLM Judge |
| **reasoning** | Quality of chain-of-thought or planning steps | 0.0 – 1.0 | LLM Judge |
| **efficiency** | Number of steps relative to optimal solution | 0.0 – 1.0 | Heuristic + LLM |

### Prompt Metrics

| Metric | Description | Score Range | Method |
|---|---|---|---|
| **correctness** | Factual accuracy of generated output | 0.0 – 1.0 | LLM Judge |
| **conciseness** | Brevity without loss of meaning | 0.0 – 1.0 | LLM Judge |
| **tone** | Adherence to desired style/voice | 0.0 – 1.0 | LLM Judge |
| **format_compliance** | Output matches required schema/format | 0 / 1 | Schema Validator |
| **instruction_following** | Response adheres to system prompt constraints | 0.0 – 1.0 | LLM Judge |

### Custom Metrics

Define your own metrics in Python:

```python
# my_metric.py
from llm_eval.metrics import Metric, MetricResult

class CostEfficiency(Metric):
    name = "cost_efficiency"
    description = "Quality-adjusted cost per query"

    def evaluate(self, sample, judge) -> MetricResult:
        quality = judge.score(sample.answer, sample.reference)
        cost = sample.metadata.get("total_cost", 0)
        score = quality / max(cost, 0.001)
        return MetricResult(score=min(score, 1.0), details={"cost": cost})
```

```bash
llm-eval run --config evals.yaml --custom-metrics my_metric.py
```

---

## ⚙️ Configuration

Full configuration reference for `evals.yaml`:

```yaml
judge:
  model: gpt-4o              # gpt-4o | claude-3-opus | gemini-pro | any OpenAI-compatible
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
    type: rag                 # rag | agent | prompt
    dataset: data.jsonl
    metrics: [faithfulness, answer_relevancy]
    threshold: 0.8            # override default
    metadata:
      version: "v2.1"
      environment: "staging"
```

---

## 🗂️ Dataset Formats

### RAG (`rag`)

```jsonl
{
  "query": "user question",
  "context": ["retrieved chunk 1", "retrieved chunk 2"],
  "answer": "generated answer",
  "reference": "ground truth answer (optional)"
}
```

### Agent (`agent`)

```jsonl
{
  "task": "Book a flight from NYC to London for next Friday",
  "tools_called": [
    {"name": "search_flights", "args": {"from": "NYC", "to": "LON", "date": "2025-01-17"}},
    {"name": "book_flight", "args": {"id": "FL123"}}
  ],
  "final_output": "Your flight FL123 is booked.",
  "expected_tools": ["search_flights", "book_flight"],
  "reference": "Flight booked successfully"
}
```

### Prompt (`prompt`)

```jsonl
{
  "input": "Summarize this article: ...",
  "output": "The article discusses...",
  "reference": "Expected summary (optional)",
  "system_prompt": "You are a concise summarizer.",
  "expected_format": "paragraph"
}
```

---

## 📈 Output Formats

```bash
# Terminal (default) — colored table summary
llm-eval run --config evals.yaml

# JSON — machine-readable
llm-eval run --config evals.yaml --output json > results.json

# CSV — spreadsheet-friendly
llm-eval run --config evals.yaml --output csv > results.csv

# HTML — shareable report
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

```bash
# Development setup
git clone https://github.com/your-org/llm-eval.git
cd llm-eval
pip install -e ".[dev]"
pytest
```

```bash
# Run linters
ruff check .
mypy llm_eval/
```

---

## 📜 License

[MIT](LICENSE) — use it freely in personal and commercial projects.

---

## 🙏 Acknowledgments

Inspired by [RAGAS](https://github.com/explodinggradients/ragas), [DeepEval](https://github.com/confident-ai/deepeval), and the broader LLM evaluation community. Built for developers who want **simplicity without sacrificing rigor**.

---

<div align="center">

**Built with ❤️ by [Your Org](https://github.com/your-org)**

⭐ Star this repo if you find it useful!

</div>