"""Context precision metric: measures signal-to-noise ratio in retrieved context."""

from __future__ import annotations

from llm_eval.metrics import Metric, MetricResult
from llm_eval.models import Sample


class ContextPrecisionMetric(Metric):
    """Measures how much of the retrieved context is relevant to the query.

    High precision means the retrieved context contains mostly useful information.
    Low precision means the context has a lot of irrelevant noise.
    Requires a reference answer to judge relevance.
    """

    name = "context_precision"
    description = "Signal-to-noise ratio in retrieved context"

    async def evaluate(self, sample: Sample) -> MetricResult:
        """Evaluate context precision for a sample.

        Args:
            sample: The evaluation sample containing context and query.

        Returns:
            MetricResult with precision score (0.0–1.0) and reasoning.
        """
        if not sample.context:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"reasoning": "No context provided"},
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
        """Build the judge prompt for context precision evaluation.

        Args:
            sample: The evaluation sample.

        Returns:
            Formatted prompt string for the judge.
        """
        context_text = "\n".join(
            f"  [{i+1}] {ctx}" for i, ctx in enumerate(sample.context)
        )
        ref_section = ""
        if sample.reference:
            ref_section = f"\n## Reference Answer\n{sample.reference}\n"

        return (
            "You are evaluating the precision of retrieved context for answering a question.\n\n"
            "Context precision measures how much of the retrieved context is actually relevant "
            "to answering the question. High precision = mostly relevant chunks. "
            "Low precision = lots of irrelevant noise.\n\n"
            f"## Question\n{sample.query}\n\n"
            f"## Retrieved Context Chunks\n{context_text}\n"
            f"{ref_section}\n"
            "## Instructions\n"
            "1. For each context chunk, determine if it is relevant to answering the question.\n"
            "2. Calculate precision = (relevant chunks) / (total chunks).\n"
            "3. Score from 0.0 (no relevant context) to 1.0 (all context is relevant).\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"score": <float>, "reasoning": "<brief explanation>"}'
        )
