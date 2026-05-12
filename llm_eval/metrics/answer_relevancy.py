"""Answer relevancy metric: measures how well the answer addresses the original query."""

from __future__ import annotations

from llm_eval.metrics import Metric, MetricResult
from llm_eval.models import Sample


class AnswerRelevancyMetric(Metric):
    """Measures how relevant the generated answer is to the original query.

    A high score indicates the answer directly and completely addresses
    the question asked. A low score means the answer is off-topic or
    only partially addresses the query.
    """

    name = "answer_relevancy"
    description = "How well the answer addresses the original query"

    async def evaluate(self, sample: Sample) -> MetricResult:
        """Evaluate answer relevancy for a sample.

        Args:
            sample: The evaluation sample.

        Returns:
            MetricResult with relevancy score (0.0–1.0) and reasoning.
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
        """Build the judge prompt for answer relevancy evaluation.

        Args:
            sample: The evaluation sample.

        Returns:
            Formatted prompt string for the judge.
        """
        return (
            "You are evaluating how relevant an answer is to the original question.\n\n"
            "Relevancy measures whether the answer directly and completely addresses "
            "what was asked. Consider:\n"
            "- Does the answer address the main question?\n"
            "- Does it provide complete information?\n"
            "- Is there irrelevant or off-topic content?\n\n"
            f"## Question\n{sample.query}\n\n"
            f"## Answer to Evaluate\n{sample.answer}\n\n"
            "## Instructions\n"
            "Score from 0.0 (completely irrelevant) to 1.0 (perfectly relevant).\n\n"
            "Respond with ONLY a JSON object:\n"
            '{"score": <float>, "reasoning": "<brief explanation>"}'
        )
