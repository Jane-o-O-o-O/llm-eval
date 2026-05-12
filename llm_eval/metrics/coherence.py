"""Coherence metric: measures answer quality — structure, fluency, and logical flow."""

from __future__ import annotations

from llm_eval.metrics import Metric, MetricResult
from llm_eval.models import Sample


class CoherenceMetric(Metric):
    """Measures the overall quality and coherence of a generated answer.

    Evaluates structural clarity, logical flow, fluency, and readability.
    Unlike faithfulness (context consistency) or relevancy (query relevance),
    coherence focuses on the intrinsic quality of the answer itself.

    Useful for catching answers that are factually correct but poorly written,
    disjointed, or hard to follow.
    """

    name = "coherence"
    description = "Answer quality: structure, fluency, and logical flow"

    async def evaluate(self, sample: Sample) -> MetricResult:
        """Evaluate coherence of a sample's answer.

        Args:
            sample: The evaluation sample.

        Returns:
            MetricResult with coherence score (0.0–1.0) and reasoning.
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
        """Build the judge prompt for coherence evaluation.

        Args:
            sample: The evaluation sample.

        Returns:
            Formatted prompt string for the judge.
        """
        return (
            "You are evaluating the coherence and quality of a generated answer.\n\n"
            "Coherence measures the intrinsic quality of the answer regardless of "
            "whether it is factually correct. Consider:\n"
            "- Structure: Is the answer well-organized with clear paragraphs/sections?\n"
            "- Fluency: Is the language natural, grammatically correct, and readable?\n"
            "- Logical flow: Do ideas connect smoothly? Are transitions clear?\n"
            "- Completeness: Does the answer feel complete and well-rounded?\n"
            "- Conciseness: Is there unnecessary repetition or filler content?\n\n"
            f"## Question\n{sample.query}\n\n"
            f"## Answer to Evaluate\n{sample.answer}\n\n"
            "## Instructions\n"
            "Score from 0.0 (incoherent, poorly written) to 1.0 (perfectly coherent and well-structured).\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"score": <float>, "reasoning": "<brief explanation>"}'
        )
