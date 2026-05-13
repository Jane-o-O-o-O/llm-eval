"""Tests for the metric framework base class and registry."""

import pytest

from llm_eval.metrics import Metric, MetricRegistry, MetricResult
from llm_eval.models import Sample


class DummyMetric(Metric):
    """A dummy metric for testing the base class."""

    name = "dummy"
    description = "A test metric"

    async def evaluate(self, sample: Sample) -> MetricResult:
        return MetricResult(name=self.name, score=1.0, details={"test": True})


class TestMetric:
    """Tests for the Metric abstract base class."""

    def test_cannot_instantiate_abstract(self) -> None:
        with pytest.raises(TypeError):
            Metric()  # type: ignore[abstract]

    def test_dummy_metric_name(self) -> None:
        metric = DummyMetric()
        assert metric.name == "dummy"

    @pytest.mark.asyncio
    async def test_dummy_metric_evaluate(self) -> None:
        metric = DummyMetric()
        sample = Sample(query="q", context=["c"], answer="a")
        result = await metric.evaluate(sample)
        assert result.score == 1.0
        assert result.name == "dummy"
        assert result.details == {"test": True}

    def test_judge_call_method_exists(self) -> None:
        """Base class should provide _judge_call for subclasses."""
        metric = DummyMetric()
        assert hasattr(metric, "_judge_call")
        assert callable(metric._judge_call)

    def test_accepts_judge_config(self) -> None:
        """Metric should accept optional JudgeConfig."""
        from llm_eval.models import JudgeConfig

        config = JudgeConfig(model="test-model")
        metric = DummyMetric(judge_config=config)
        assert metric._judge_config == config

    def test_default_judge_config_is_none(self) -> None:
        metric = DummyMetric()
        assert metric._judge_config is None


class TestMetricRegistry:
    """Tests for the metric registry."""

    def test_register_metric(self) -> None:
        registry = MetricRegistry()
        registry.register(DummyMetric())
        assert "dummy" in registry

    def test_get_metric(self) -> None:
        registry = MetricRegistry()
        registry.register(DummyMetric())
        metric = registry.get("dummy")
        assert isinstance(metric, DummyMetric)

    def test_get_unregistered_raises(self) -> None:
        registry = MetricRegistry()
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent")

    def test_list_metrics(self) -> None:
        registry = MetricRegistry()
        registry.register(DummyMetric())
        names = registry.list_metrics()
        assert names == ["dummy"]

    def test_is_registered(self) -> None:
        registry = MetricRegistry()
        assert not registry.is_registered("dummy")
        registry.register(DummyMetric())
        assert registry.is_registered("dummy")


class TestDefaultRegistry:
    """Tests for the default metric registry."""

    def test_has_all_builtin_metrics(self) -> None:
        from llm_eval.metrics import get_default_registry

        registry = get_default_registry()
        expected = {
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "format_compliance",
            "toxicity",
            "answer_correctness",
            "coherence",
        }
        names = set(registry.list_metrics())
        assert expected.issubset(names)

    def test_registry_with_judge_config(self) -> None:
        from llm_eval.metrics import get_default_registry
        from llm_eval.models import JudgeConfig

        config = JudgeConfig(model="custom-model")
        registry = get_default_registry(judge_config=config)
        metric = registry.get("faithfulness")
        assert metric._judge_config == config

    def test_coherence_is_registered(self) -> None:
        from llm_eval.metrics import get_default_registry

        registry = get_default_registry()
        assert "coherence" in registry
        metric = registry.get("coherence")
        assert metric.description != ""
