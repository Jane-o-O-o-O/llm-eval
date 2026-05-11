"""Context recall metric: measures coverage of reference by retrieved context."""

from __future__ import annotations

from typing import Any

from llm_eval.metrics import Metric, MetricResult
from llm_eval.models import Sample


class ContextRecallMetric(Metric):
    """Measures how well the retrieved context covers the reference answer.

    High recall means the context contains the information needed to produce
    the reference answer. Low recall means important information is missing.
    Requires a reference answer for comparison.
    """

    name = "context_recall"
    description = "Coverage of reference answer by retrieved context"

    async def evaluate(self, sample: Sample) -> MetricResult:
        """Evaluate context recall for a sample.

        Args:
            sample: The evaluation sample containing context and reference.

        Returns:
            MetricResult with recall score (0.0–1.0) and reasoning.
        """
        if not sample.context:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reasoning": "No context provided"},
            )

        if not sample.reference:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reasoning": "Reference answer required for context recall"},
            )

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
        """Build the judge prompt for context recall evaluation.

        Args:
            sample: The evaluation sample.

        Returns:
            Formatted prompt string for the judge.
        """
        context_text = "\n".join(
            f"  [{i+1}] {ctx}" for i, ctx in enumerate(sample.context)
        )
        return (
            "You are evaluating the recall of retrieved context for answering a question.\n\n"
            "Context recall measures whether the retrieved context contains the information "
            "needed to produce the reference answer. High recall = the context covers the "
            "reference answer well. Low recall = important information is missing.\n\n"
            f"## Question\n{sample.query}\n\n"
            f"## Retrieved Context Chunks\n{context_text}\n\n"
            f"## Reference Answer\n{sample.reference}\n\n"
            "## Instructions\n"
            "1. Break the reference answer into individual claims/facts.\n"
            "2. Check which claims can be supported by the retrieved context.\n"
            "3. Calculate recall = (supported claims) / (total claims).\n"
            "4. Score from 0.0 (context covers nothing) to 1.0 (context covers everything).\n\n"
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
        from llm_eval.judge import Judge

        judge = Judge()
        return await judge.call(prompt)
