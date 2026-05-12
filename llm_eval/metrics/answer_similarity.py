"""Answer similarity metric: semantic similarity between answer and reference."""

from __future__ import annotations

from llm_eval.metrics import Metric, MetricResult
from llm_eval.models import Sample


class AnswerSimilarityMetric(Metric):
    """Measures semantic similarity between the answer and a reference answer.

    Uses an LLM judge to compare the generated answer against the ground-truth
    reference. Unlike answer_correctness (which uses token overlap + LLM),
    this is purely LLM-judge based and focuses on meaning overlap.

    Requires a reference answer in the sample. If no reference is provided,
    returns a score of 0.0 with a warning.
    """

    name = "answer_similarity"
    description = "Semantic similarity between answer and reference answer"

    async def evaluate(self, sample: Sample) -> MetricResult:
        """Evaluate semantic similarity of answer to reference.

        Args:
            sample: The evaluation sample (must have reference).

        Returns:
            MetricResult with similarity score (0.0–1.0) and reasoning.
        """
        if not sample.reference:
            return MetricResult(
                name=self.name,
                score=0.0,
                details={"warning": "No reference answer provided"},
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
        """Build the judge prompt for similarity evaluation.

        Args:
            sample: The evaluation sample.

        Returns:
            Formatted prompt string for the judge.
        """
        return (
            "You are evaluating the semantic similarity between a generated answer "
            "and a reference answer. Focus on whether they convey the same meaning, "
            "even if worded differently.\n\n"
            f"## Question\n{sample.query}\n\n"
            f"## Reference Answer\n{sample.reference}\n\n"
            f"## Generated Answer\n{sample.answer}\n\n"
            "## Instructions\n"
            "1. Compare the key information in both answers.\n"
            "2. Consider paraphrasing, synonyms, and different phrasing as equivalent.\n"
            "3. Score from 0.0 (completely different meaning) to 1.0 (identical meaning).\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"score": <float>, "reasoning": "<brief explanation>"}'
        )
