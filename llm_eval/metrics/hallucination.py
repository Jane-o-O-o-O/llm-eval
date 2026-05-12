"""Hallucination metric: detects fabricated claims not supported by context."""

from __future__ import annotations

from llm_eval.metrics import Metric, MetricResult
from llm_eval.models import Sample


class HallucinationMetric(Metric):
    """Measures the degree of hallucination in the generated answer.

    This metric uses an LLM judge to identify claims in the answer that are
    NOT supported by the retrieved context. A low hallucination score means
    the answer is well-grounded; a high score means it contains fabricated info.

    Score interpretation: 0.0 = no hallucination, 1.0 = fully hallucinated.
    (Inverted from faithfulness for clarity.)
    """

    name = "hallucination"
    description = "Degree of fabricated claims not supported by context"

    async def evaluate(self, sample: Sample) -> MetricResult:
        """Evaluate hallucination in a sample's answer against its context.

        Args:
            sample: The evaluation sample containing context and answer.

        Returns:
            MetricResult with hallucination score (0.0–1.0) and details.
        """
        prompt = self._build_prompt(sample)
        response = await self._judge_call(prompt)
        score = float(response.get("score", 0.0))
        claims = response.get("hallucinated_claims", [])
        return MetricResult(
            name=self.name,
            score=max(0.0, min(1.0, score)),
            details={
                "reasoning": response.get("reasoning", ""),
                "hallucinated_claims": claims,
            },
        )

    @staticmethod
    def _build_prompt(sample: Sample) -> str:
        """Build the judge prompt for hallucination detection.

        Args:
            sample: The evaluation sample.

        Returns:
            Formatted prompt string for the judge.
        """
        context_text = "\n".join(f"- {ctx}" for ctx in sample.context)
        return (
            "You are detecting hallucinations in an answer. "
            "A hallucination is a claim or fact that CANNOT be inferred from the provided context.\n\n"
            f"## Question\n{sample.query}\n\n"
            f"## Retrieved Context\n{context_text}\n\n"
            f"## Answer to Evaluate\n{sample.answer}\n\n"
            "## Instructions\n"
            "1. Identify each distinct claim/fact in the answer.\n"
            "2. For each claim, determine if it is supported by the context.\n"
            "3. List any claims that are fabricated or unsupported.\n"
            "4. Score from 0.0 (no hallucination) to 1.0 (completely hallucinated).\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"score": <float>, "hallucinated_claims": ["claim1", "claim2"], "reasoning": "<explanation>"}'
        )
