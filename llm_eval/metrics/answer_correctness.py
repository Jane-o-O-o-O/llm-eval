"""Answer correctness metric: hybrid token-overlap + LLM-judge evaluation."""

from __future__ import annotations

import re
from typing import Any

from llm_eval.metrics import Metric, MetricResult
from llm_eval.models import Sample


def _tokenize(text: str) -> set[str]:
    """Lowercase and split text into alphanumeric tokens.

    Args:
        text: Input text to tokenize.

    Returns:
        Set of lowercase tokens.
    """
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def token_overlap_score(answer: str, reference: str) -> float:
    """Compute token-level F1 overlap between answer and reference.

    Uses the F1 of precision (tokens in answer that are in reference)
    and recall (tokens in reference that are in answer).

    Args:
        answer: The generated answer text.
        reference: The ground truth reference text.

    Returns:
        F1 score between 0.0 and 1.0.
    """
    answer_tokens = _tokenize(answer)
    reference_tokens = _tokenize(reference)

    if not answer_tokens and not reference_tokens:
        return 1.0
    if not answer_tokens or not reference_tokens:
        return 0.0

    common = answer_tokens & reference_tokens
    if not common:
        return 0.0

    precision = len(common) / len(answer_tokens)
    recall = len(common) / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


class AnswerCorrectnessMetric(Metric):
    """Measures answer correctness using a hybrid approach.

    Combines token-level F1 overlap with an LLM judge evaluation.
    When a reference answer is available, computes a weighted blend:
    - 40% token overlap (fast, deterministic)
    - 60% LLM judge semantic correctness

    Falls back to LLM-only when no reference is provided.
    """

    name = "answer_correctness"
    description = "Hybrid token-overlap + LLM-judge correctness against reference"

    def __init__(self, token_weight: float = 0.4, judge_weight: float = 0.6, **kwargs: Any) -> None:
        """Initialize with configurable blend weights.

        Args:
            token_weight: Weight for the token overlap component.
            judge_weight: Weight for the LLM judge component.
            **kwargs: Passed to base Metric (e.g., judge_config).
        """
        super().__init__(**kwargs)
        self.token_weight = token_weight
        self.judge_weight = judge_weight

    async def evaluate(self, sample: Sample) -> MetricResult:
        """Evaluate answer correctness.

        Args:
            sample: The evaluation sample (must have `reference` for token overlap).

        Returns:
            MetricResult with correctness score (0.0–1.0) and details.
        """
        judge_response = await self._judge_call(self._build_prompt(sample))
        judge_score = float(judge_response.get("score", 0.0))

        if sample.reference:
            tok_score = token_overlap_score(sample.answer, sample.reference)
            blended = self.token_weight * tok_score + self.judge_weight * judge_score
            return MetricResult(
                name=self.name,
                score=max(0.0, min(1.0, blended)),
                details={
                    "token_overlap": round(tok_score, 4),
                    "judge_score": round(judge_score, 4),
                    "reasoning": judge_response.get("reasoning", ""),
                },
            )

        # No reference — judge-only
        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, judge_score)),
            details={
                "token_overlap": None,
                "judge_score": round(judge_score, 4),
                "reasoning": judge_response.get("reasoning", ""),
            },
        )

    @staticmethod
    def _build_prompt(sample: Sample) -> str:
        """Build the judge prompt for correctness evaluation.

        Args:
            sample: The evaluation sample.

        Returns:
            Formatted prompt string for the judge.
        """
        ref_section = ""
        if sample.reference:
            ref_section = f"\n## Reference Answer\n{sample.reference}\n"

        return (
            "You are evaluating the correctness of an answer to a question.\n\n"
            "Correctness measures whether the answer provides accurate, complete, "
            "and factually correct information.\n\n"
            f"## Question\n{sample.query}\n\n"
            f"## Answer to Evaluate\n{sample.answer}\n"
            f"{ref_section}\n"
            "## Instructions\n"
            "Score from 0.0 (completely incorrect) to 1.0 (fully correct).\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"score": <float>, "reasoning": "<brief explanation>"}'
        )
