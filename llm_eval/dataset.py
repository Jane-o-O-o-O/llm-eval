"""Dataset loader for JSONL evaluation files."""

from __future__ import annotations

import json

from llm_eval.models import Sample

REQUIRED_FIELDS = {"query", "context", "answer"}


def load_jsonl(filepath: str) -> list[Sample]:
    """Load evaluation samples from a JSONL file.

    Each line should be a JSON object with at minimum: query, context, answer.
    Optional fields: reference, metadata.

    Args:
        filepath: Path to the JSONL file.

    Returns:
        List of Sample objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line has invalid JSON or missing required fields.
    """
    samples: list[Sample] = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_num}: {exc}"
                ) from exc

            missing = REQUIRED_FIELDS - set(data.keys())
            if missing:
                raise ValueError(
                    f"Missing required field(s) {missing} on line {line_num}"
                )

            samples.append(Sample.from_dict(data))

    return samples
