"""Context recall metric: measures how much of the reference is covered by retrieved context."""

from __future__ import annotations

from typing import Any

from llm_eval.metrics import Metric, MetricResult
from llm_eval.models import Sample


class ContextRecallMetric(Metric):
    """Measures how well the retrieved context covers the ground-truth reference.

    This metric uses an LLM judge to identify claims in the reference answer,
    then checks what fraction of those claims can be inferred from the retrieved
    context. A high score means the context contains most of the information
    needed to answer the question correctly.
    """

    name = "context_recall"
    description = "How much of the ground-truth reference is supported by the context"

    async def evaluate(self, sample: Sample) -> MetricResult:
        """Evaluate context recall for a sample.

        Args:
            sample: The evaluation sample with context and reference.

        Returns:
            MetricResult with context recall score (0.0–1.0) and reasoning.
        """
        if not sample.reference:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reasoning": "No reference answer provided; cannot compute recall."},
            )

        prompt = self._build_prompt(sample)
        response = await self._judge_call(prompt)
        score = float(response.get("score", 0.0))
        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, score)),
            details={
                "reasoning": response.get("reasoning", ""),
                "claims_supported": response.get("claims_supported", 0),
                "claims_total": response.get("claims_total", 0),
            },
        )

    @staticmethod
    def _build_prompt(sample: Sample) -> str:
        """Build the judge prompt for context recall evaluation.

        Args:
            sample: The evaluation sample.

        Returns:
            Formatted prompt string for the judge.
        """
        context_text = "\n".join(f"- {ctx}" for ctx in sample.context)

        return (
            "You are evaluating context recall — whether the retrieved context "
            "contains the information needed to produce the ground-truth reference answer.\n\n"
            f"## Question\n{sample.query}\n\n"
            f"## Retrieved Context\n{context_text}\n\n"
            f"## Ground-Truth Reference Answer\n{sample.reference}\n\n"
            "## Instructions\n"
            "1. Break the reference answer into individual factual claims.\n"
            "2. Check if each claim can be inferred from the retrieved context.\n"
            "3. Score = (number of supported claims) / (total claims).\n"
            "Score from 0.0 (no claims supported) to 1.0 (all claims supported).\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"score": <float>, "reasoning": "<brief explanation>", '
            '"claims_supported": <int>, "claims_total": <int>}\n'
        )

    async def _judge_call(self, prompt: str) -> dict[str, Any]:
        """Call the LLM judge. Override in tests for mocking.

        Args:
            prompt: The prompt to send to the judge.

        Returns:
            Parsed JSON response from the judge.
        """
        from llm_eval.judge import Judge

        judge = Judge()
        return await judge.call(prompt)
