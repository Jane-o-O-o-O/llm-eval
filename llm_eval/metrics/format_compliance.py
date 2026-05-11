"""Format compliance metric: rule-based format checking without LLM judge."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from llm_eval.metrics import Metric, MetricResult
from llm_eval.models import Sample


@dataclass
class FormatComplianceMetric(Metric):
    """Checks whether the generated answer complies with format requirements.

    This is a rule-based metric that does NOT use an LLM judge. It checks:
    - Maximum answer length
    - Forbidden patterns (phrases that should not appear)
    - Required sections (keywords that must be present)

    Each failed check reduces the score proportionally.
    """

    name: str = "format_compliance"
    description: str = "Rule-based format compliance checking (length, forbidden patterns, required sections)"
    max_length: int | None = None
    forbidden_patterns: list[str] = field(default_factory=list)
    required_sections: list[str] = field(default_factory=list)

    async def evaluate(self, sample: Sample) -> MetricResult:
        """Evaluate format compliance of a sample's answer.

        Args:
            sample: The evaluation sample.

        Returns:
            MetricResult with compliance score (0.0–1.0) and details.
        """
        violations = 0
        total_checks = 0
        length_penalty = 0
        matched_forbidden: list[str] = []
        missing_sections: list[str] = []

        answer_lower = sample.answer.lower()

        # Check max length
        if self.max_length is not None:
            total_checks += 1
            if len(sample.answer) > self.max_length:
                violations += 1
                length_penalty = 1

        # Check forbidden patterns
        if self.forbidden_patterns:
            total_checks += 1
            for pattern in self.forbidden_patterns:
                if pattern.lower() in answer_lower:
                    matched_forbidden.append(pattern)
            if matched_forbidden:
                violations += 1

        # Check required sections
        if self.required_sections:
            total_checks += 1
            for section in self.required_sections:
                if section.lower() not in answer_lower:
                    missing_sections.append(section)
            if missing_sections:
                violations += 1

        # If no checks configured, score is 1.0
        if total_checks == 0:
            score = 1.0
        else:
            score = (total_checks - violations) / total_checks

        return MetricResult(
            name=self.name,
            score=score,
            details={
                "length_penalty": length_penalty,
                "matched_forbidden": matched_forbidden,
                "missing_sections": missing_sections,
                "total_checks": total_checks,
                "violations": violations,
            },
        )
