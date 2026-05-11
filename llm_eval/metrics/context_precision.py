"""Context precision metric: measures whether relevant context chunks are ranked higher."""

from __future__ import annotations

from typing import Any

from llm_eval.metrics import Metric, MetricResult
from llm_eval.models import Sample


class ContextPrecisionMetric(Metric):
    """Measures how well the retrieved context chunks are ranked by relevance.

    This metric uses an LLM judge to determine which context chunks are
    relevant to answering the query, then computes a weighted precision score
    where relevant chunks ranked higher (earlier) score better. This rewards
    retrieval systems that put the most useful context first.

    The score is computed as Mean Average Precision (MAP):
        score = sum(precision@k for k where chunk k is relevant) / num_relevant_chunks
    """

    name = "context_precision"
    description = "Whether relevant context chunks are ranked higher in the retrieval"

    async def evaluate(self, sample: Sample) -> MetricResult:
        """Evaluate context precision for a sample.

        Args:
            sample: The evaluation sample containing context and reference.

        Returns:
            MetricResult with context precision score (0.0–1.0) and reasoning.
        """
        prompt = self._build_prompt(sample)
        response = await self._judge_call(prompt)
        score = float(response.get("score", 0.0))
        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, score)),
            details={
                "reasoning": response.get("reasoning", ""),
                "relevant_positions": response.get("relevant_positions", []),
            },
        )

    @staticmethod
    def _build_prompt(sample: Sample) -> str:
        """Build the judge prompt for context precision evaluation.

        Args:
            sample: The evaluation sample.

        Returns:
            Formatted prompt string for the judge.
        """
        chunks_text = "\n".join(
            f"[{i + 1}] {ctx}" for i, ctx in enumerate(sample.context)
        )
        reference_text = sample.reference or sample.answer

        return (
            "You are evaluating context precision — whether the most relevant "
            "context chunks are ranked higher (earlier) in the retrieval list.\n\n"
            f"## Question\n{sample.query}\n\n"
            f"## Retrieved Context (ordered by retrieval rank)\n{chunks_text}\n\n"
            f"## Reference Answer\n{reference_text}\n\n"
            "## Instructions\n"
            "1. Determine which context chunks (by number) are relevant to answering the question.\n"
            "2. Score from 0.0 (no relevant chunks or all relevant chunks at the bottom) "
            "to 1.0 (all relevant chunks at the top).\n"
            "3. Use Mean Average Precision: for each relevant chunk at position k, "
            "compute precision@k = (relevant chunks in top k) / k, then average.\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"score": <float>, "reasoning": "<brief explanation>", '
            '"relevant_positions": [<list of 1-indexed positions of relevant chunks>]}\n'
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
