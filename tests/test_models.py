"""Tests for core data models."""

from llm_eval.models import EvalResult, MetricResult, Sample


class TestSample:
    """Tests for the Sample data model."""

    def test_create_rag_sample(self) -> None:
        sample = Sample(
            query="What is Python?",
            context=["Python is a programming language."],
            answer="Python is a popular programming language.",
            reference="Python is a programming language.",
        )
        assert sample.query == "What is Python?"
        assert len(sample.context) == 1
        assert sample.answer == "Python is a popular programming language."
        assert sample.reference == "Python is a programming language."

    def test_sample_without_reference(self) -> None:
        sample = Sample(
            query="What is Python?",
            context=["Python is a programming language."],
            answer="Python is a programming language.",
        )
        assert sample.reference is None

    def test_sample_with_metadata(self) -> None:
        sample = Sample(
            query="test",
            context=["ctx"],
            answer="ans",
            metadata={"version": "v1"},
        )
        assert sample.metadata == {"version": "v1"}

    def test_sample_from_dict(self) -> None:
        data = {
            "query": "What is Python?",
            "context": ["Python is a programming language."],
            "answer": "Python is a programming language.",
            "reference": "Python is a programming language.",
        }
        sample = Sample.from_dict(data)
        assert sample.query == "What is Python?"
        assert sample.reference == "Python is a programming language."

    def test_sample_from_dict_minimal(self) -> None:
        data = {"query": "test", "context": ["ctx"], "answer": "ans"}
        sample = Sample.from_dict(data)
        assert sample.reference is None
        assert sample.metadata == {}


class TestMetricResult:
    """Tests for MetricResult data model."""

    def test_create_metric_result(self) -> None:
        result = MetricResult(name="faithfulness", score=0.95)
        assert result.name == "faithfulness"
        assert result.score == 0.95
        assert result.details == {}

    def test_metric_result_with_details(self) -> None:
        result = MetricResult(
            name="faithfulness",
            score=0.8,
            details={"reasoning": "Mostly faithful"},
        )
        assert result.details["reasoning"] == "Mostly faithful"

    def test_metric_result_score_bounds(self) -> None:
        result = MetricResult(name="test", score=0.0)
        assert result.score == 0.0
        result = MetricResult(name="test", score=1.0)
        assert result.score == 1.0

    def test_metric_result_to_dict(self) -> None:
        result = MetricResult(name="test", score=0.5, details={"key": "val"})
        d = result.to_dict()
        assert d == {"name": "test", "score": 0.5, "details": {"key": "val"}}


class TestEvalResult:
    """Tests for EvalResult data model."""

    def test_create_eval_result(self) -> None:
        metrics = [MetricResult(name="faithfulness", score=0.9)]
        result = EvalResult(sample_index=0, metrics=metrics)
        assert result.sample_index == 0
        assert len(result.metrics) == 1
        assert result.metrics[0].name == "faithfulness"

    def test_overall_score(self) -> None:
        metrics = [
            MetricResult(name="a", score=0.8),
            MetricResult(name="b", score=0.6),
        ]
        result = EvalResult(sample_index=0, metrics=metrics)
        assert result.overall_score == 0.7

    def test_overall_score_empty(self) -> None:
        result = EvalResult(sample_index=0, metrics=[])
        assert result.overall_score == 0.0

    def test_to_dict(self) -> None:
        metrics = [MetricResult(name="faithfulness", score=0.9)]
        result = EvalResult(sample_index=0, metrics=metrics)
        d = result.to_dict()
        assert d["sample_index"] == 0
        assert d["overall_score"] == 0.9
        assert len(d["metrics"]) == 1
