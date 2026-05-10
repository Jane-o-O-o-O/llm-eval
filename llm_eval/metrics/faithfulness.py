"""Faithfulness metric: measures factual consistency between answer and retrieved context."""

from __future__ import annotations

from typing import Any

from llm_eval.metrics import Metric, MetricResult
from llm_eval.models import Sample


class FaithfulnessMetric(Metric):
    """Measures whether the generated answer is supported by the retrieved context.

    This metric uses an LLM judge to verify that each claim in the answer
    can be inferred from the provided context chunks. A high score indicates
    the answer does not hallucinate information not present in the context.
    """

    name = "faithfulness"
    description = "Factual consistency between answer and retrieved context"

    async def evaluate(self, sample: Sample) -> MetricResult:
        """Evaluate faithfulness of a sample's answer against its context.

        Args:
            sample: The evaluation sample containing context and answer.

        Returns:
            MetricResult with faithfulness score (0.0–1.0) and reasoning.
        """
        prompt = self._build_prompt(sample)
        response = await self._judge_call(prompt)
        score = float(response.get("score", 0.0))
        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, score)),
            details={"reasoning": response.get("reasoning", "")},
        )

    @staticmethod
    def _build_prompt(sample: Sample) -> str:
        """Build the judge prompt for faithfulness evaluation.

        Args:
            sample: The evaluation sample.

        Returns:
            Formatted prompt string for the judge.
        """
        context_text = "\n".join(f"- {ctx}" for ctx in sample.context)
        return (
            "You are evaluating the faithfulness of an answer to its source context.\n\n"
            "Faithfulness measures whether the answer's claims are supported by the context. "
            "The answer should not contain information that cannot be inferred from the context.\n\n"
            f"## Question\n{sample.query}\n\n"
            f"## Retrieved Context\n{context_text}\n\n"
            f"## Answer to Evaluate\n{sample.answer}\n\n"
            "## Instructions\n"
            "1. Identify each claim/fact in the answer.\n"
            "2. Check if each claim is supported by the context.\n"
            "3. Score from 0.0 (completely unfaithful) to 1.0 (fully faithful).\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"score": <float>, "reasoning": "<brief explanation>"}'
        )

    async def _judge_call(self, prompt: str) -> dict[str, Any]:
        """Call the LLM judge. Override in tests for mocking.

        Args:
            prompt: The prompt to send to the judge.

        Returns:
            Parsed JSON response from the judge.
        """
        # Lazy import to avoid circular dependency and allow easier testing
        from llm_eval.judge import Judge

        judge = Judge()
        return await judge.call(prompt)
