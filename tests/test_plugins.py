"""Tests for the plugin loader."""

from __future__ import annotations

import sys
import types

import pytest

from llm_eval.metrics import Metric, MetricRegistry, MetricResult
from llm_eval.models import Sample
from llm_eval.plugins import load_custom_metrics


class DummyMetric(Metric):
    """A test metric for plugin loading tests."""

    name = "dummy_custom"
    description = "A dummy custom metric for testing"

    async def evaluate(self, sample: Sample) -> MetricResult:
        return MetricResult(name=self.name, score=0.5)


class AnotherMetric(Metric):
    """Another test metric for plugin loading tests."""

    name = "another_custom"
    description = "Another test metric"

    async def evaluate(self, sample: Sample) -> MetricResult:
        return MetricResult(name=self.name, score=0.7)


class NotAMetric:
    """A class that is NOT a Metric subclass."""


@pytest.fixture(autouse=True)
def _cleanup_modules():
    """Ensure fake modules are cleaned up after each test."""
    before = set(sys.modules.keys())
    yield
    after = set(sys.modules.keys())
    for mod_name in after - before:
        del sys.modules[mod_name]


class TestLoadCustomMetrics:
    """Tests for load_custom_metrics."""

    def _make_fake_module(self, name: str, **attrs) -> types.ModuleType:
        """Create a fake module with the given attributes."""
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    def test_load_valid_metric(self) -> None:
        registry = MetricRegistry()
        self._make_fake_module("test_plugin_valid", DummyMetric=DummyMetric)
        specs = [{"module": "test_plugin_valid", "class": "DummyMetric"}]
        loaded = load_custom_metrics(registry, specs)
        assert "dummy_custom" in loaded
        assert registry.is_registered("dummy_custom")

    def test_load_non_metric_class_raises(self) -> None:
        registry = MetricRegistry()
        self._make_fake_module("test_plugin_not_metric", NotAMetric=NotAMetric)
        specs = [{"module": "test_plugin_not_metric", "class": "NotAMetric"}]
        with pytest.raises(TypeError, match="must be a subclass"):
            load_custom_metrics(registry, specs)

    def test_missing_module_raises(self) -> None:
        registry = MetricRegistry()
        specs = [{"module": "nonexistent_module_xyz", "class": "Foo"}]
        with pytest.raises(ImportError):
            load_custom_metrics(registry, specs)

    def test_missing_class_raises(self) -> None:
        registry = MetricRegistry()
        self._make_fake_module("test_plugin_empty")
        specs = [{"module": "test_plugin_empty", "class": "Nonexistent"}]
        with pytest.raises(AttributeError):
            load_custom_metrics(registry, specs)

    def test_skip_entry_with_missing_fields(self) -> None:
        registry = MetricRegistry()
        specs = [{"module": ""}, {"class": ""}, {}]
        loaded = load_custom_metrics(registry, specs)
        assert loaded == []

    def test_load_multiple_metrics(self) -> None:
        registry = MetricRegistry()
        self._make_fake_module(
            "test_plugin_multi",
            DummyMetric=DummyMetric,
            AnotherMetric=AnotherMetric,
        )
        specs = [
            {"module": "test_plugin_multi", "class": "DummyMetric"},
            {"module": "test_plugin_multi", "class": "AnotherMetric"},
        ]
        loaded = load_custom_metrics(registry, specs)
        assert len(loaded) == 2
        assert "dummy_custom" in loaded
        assert "another_custom" in loaded
