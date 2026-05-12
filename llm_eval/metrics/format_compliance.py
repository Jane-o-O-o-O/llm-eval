"""Format compliance metric: checks if output matches expected format/schema."""

from __future__ import annotations

import json
import re
from typing import Any

from llm_eval.metrics import Metric, MetricResult
from llm_eval.models import Sample


def _validate_json(text: str) -> tuple[bool, str]:
    """Validate that text is valid JSON."""
    try:
        json.loads(text)
        return True, "Valid JSON"
    except json.JSONDecodeError as exc:
        return False, f"Invalid JSON: {exc}"


def _validate_markdown_heading(text: str) -> tuple[bool, str]:
    """Validate that text starts with a markdown heading."""
    if re.match(r"^#{1,6}\s+", text.strip()):
        return True, "Starts with markdown heading"
    return False, "Does not start with a markdown heading"


def _validate_bullet_list(text: str) -> tuple[bool, str]:
    """Validate that text contains a bullet list."""
    lines = text.strip().split("\n")
    bullet_lines = [l for l in lines if re.match(r"^\s*[-*+]\s+", l)]
    if len(bullet_lines) >= 2:
        return True, f"Contains {len(bullet_lines)} bullet items"
    return False, "Does not contain a proper bullet list (need >= 2 items)"


def _validate_numbered_list(text: str) -> tuple[bool, str]:
    """Validate that text contains a numbered list."""
    lines = text.strip().split("\n")
    numbered_lines = [l for l in lines if re.match(r"^\s*\d+[.)]\s+", l)]
    if len(numbered_lines) >= 2:
        return True, f"Contains {len(numbered_lines)} numbered items"
    return False, "Does not contain a proper numbered list (need >= 2 items)"


# Built-in format validators
_BUILTINS: dict[str, Any] = {
    "json": _validate_json,
    "markdown_heading": _validate_markdown_heading,
    "bullet_list": _validate_bullet_list,
    "numbered_list": _validate_numbered_list,
}


class FormatComplianceMetric(Metric):
    """Measures whether the answer matches the expected format.

    This is a deterministic metric that does NOT require an LLM judge.
    Supported formats: json, markdown_heading, bullet_list, numbered_list.
    Also supports word count constraints via metadata (min_words, max_words).
    """

    name = "format_compliance"
    description = "Output matches required schema/format (deterministic)"

    def __init__(self, **kwargs: Any) -> None:
        """Initialize with built-in validators."""
        super().__init__(**kwargs)
        self._validators = dict(_BUILTINS)

    async def evaluate(self, sample: Sample) -> MetricResult:
        """Evaluate format compliance for a sample.

        Args:
            sample: The evaluation sample. Uses metadata['expected_format']
                    and optionally metadata['min_words']/metadata['max_words'].

        Returns:
            MetricResult with compliance score (0.0 or 1.0) and details.
        """
        expected_format = sample.metadata.get("expected_format")
        checks: list[dict[str, str | bool]] = []
        all_passed = True

        # Format validation
        if expected_format:
            if expected_format not in self._validators:
                checks.append({
                    "check": expected_format,
                    "passed": False,
                    "detail": f"Unknown format: {expected_format}",
                })
                all_passed = False
            else:
                validator = self._validators[expected_format]
                passed, detail = validator(sample.answer)
                checks.append({
                    "check": expected_format,
                    "passed": passed,
                    "detail": detail,
                })
                if not passed:
                    all_passed = False

        # Word count validation
        min_words = sample.metadata.get("min_words")
        max_words = sample.metadata.get("max_words")
        if min_words is not None or max_words is not None:
            word_count = len(sample.answer.split())
            if min_words is not None and word_count < min_words:
                checks.append({
                    "check": "min_words",
                    "passed": False,
                    "detail": f"Got {word_count} words, minimum {min_words}",
                })
                all_passed = False
            if max_words is not None and word_count > max_words:
                checks.append({
                    "check": "max_words",
                    "passed": False,
                    "detail": f"Got {word_count} words, maximum {max_words}",
                })
                all_passed = False
            if all_passed or not checks:
                checks.append({
                    "check": "word_count",
                    "passed": True,
                    "detail": f"{word_count} words within bounds",
                })

        if not checks:
            return MetricResult(
                name=self.name,
                score=1.0,
                details={"reasoning": "No format constraints specified", "checks": []},
            )

        return MetricResult(
            name=self.name,
            score=1.0 if all_passed else 0.0,
            details={
                "reasoning": "All checks passed" if all_passed else "Some checks failed",
                "checks": checks,
            },
        )
