"""Tests for the metric framework base class and registry."""

import pytest
from llm_eval.metrics import Metric, MetricResult, MetricRegistry
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

    def test_get_unknown_metric_raises(self) -> None:
        registry = MetricRegistry()
        with pytest.raises(KeyError, match="unknown_metric"):
            registry.get("unknown_metric")

    def test_list_metrics(self) -> None:
        registry = MetricRegistry()
        registry.register(DummyMetric())
        names = registry.list_metrics()
        assert "dummy" in names

    def test_duplicate_register_overwrites(self) -> None:
        registry = MetricRegistry()
        registry.register(DummyMetric())
        registry.register(DummyMetric())
        assert len(registry.list_metrics()) == 1

    def test_is_registered(self) -> None:
        registry = MetricRegistry()
        assert not registry.is_registered("dummy")
        registry.register(DummyMetric())
        assert registry.is_registered("dummy")
