"""Tests for config file extends/inheritance."""

from __future__ import annotations

import os
import tempfile

import yaml

from llm_eval.models import EvalConfig


class TestConfigExtends:
    """Test that config files can extend base configs."""

    def test_from_dict_with_extends_resolved(self):
        """EvalConfig.from_dict should work with pre-merged config."""
        # The extends logic is handled at the CLI level before from_dict.
        # This tests that from_dict accepts extended configs.
        data = {
            "judge": {"model": "gpt-4o"},
            "defaults": {"threshold": 0.8},
            "evaluations": [
                {
                    "name": "test",
                    "dataset": "data.jsonl",
                    "metrics": ["faithfulness"],
                }
            ],
        }
        config = EvalConfig.from_dict(data)
        assert config.threshold == 0.8
        assert config.judge.model == "gpt-4o"

    def test_extends_merges_metrics(self):
        """When base has metrics and child overrides, child should win for overlapping keys."""
        # Simulated merge behavior
        base = {
            "judge": {"model": "gpt-4o"},
            "defaults": {"threshold": 0.7},
            "evaluations": [
                {"name": "base-eval", "dataset": "data.jsonl", "metrics": ["faithfulness"]}
            ],
        }
        child = {
            "evaluations": [
                {"name": "child-eval", "dataset": "data.jsonl", "metrics": ["toxicity"]}
            ],
        }
        # Merge: child overrides evaluations entirely (list replacement)
        merged = {**base}
        for key, value in child.items():
            merged[key] = value
        assert len(merged["evaluations"]) == 1
        assert merged["evaluations"][0]["name"] == "child-eval"
        # Base judge defaults preserved
        assert merged["judge"]["model"] == "gpt-4o"

    def test_extends_preserves_nested_defaults(self):
        """Child config without defaults should inherit base defaults."""
        base = {
            "judge": {"model": "gpt-4o", "temperature": 0.0},
            "defaults": {"threshold": 0.8, "parallel": 3},
        }
        child = {
            "judge": {"model": "claude-3-opus"},
        }
        # Shallow merge of judge dict
        merged_judge = {**base.get("judge", {}), **child.get("judge", {})}
        assert merged_judge["model"] == "claude-3-opus"
        assert merged_judge["temperature"] == 0.0  # preserved from base

    def test_resolve_extends_function(self):
        """Test the _resolve_extends helper."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_path = os.path.join(tmpdir, "base.yaml")
            child_path = os.path.join(tmpdir, "child.yaml")

            base_config = {
                "judge": {"model": "gpt-4o"},
                "defaults": {"threshold": 0.7, "parallel": 3},
            }
            child_config = {
                "extends": "base.yaml",
                "judge": {"model": "claude-3-opus"},
                "evaluations": [
                    {"name": "test", "dataset": "data.jsonl", "metrics": ["faithfulness"]}
                ],
            }

            with open(base_path, "w") as f:
                yaml.dump(base_config, f)
            with open(child_path, "w") as f:
                yaml.dump(child_config, f)

            from llm_eval.cli import _resolve_extends

            merged = _resolve_extends(child_config, os.path.dirname(child_path))
            assert merged["judge"]["model"] == "claude-3-opus"
            assert merged["defaults"]["threshold"] == 0.7
            assert merged["defaults"]["parallel"] == 3
            assert "extends" not in merged  # extends key removed

    def test_resolve_extends_no_extends_key(self):
        """Config without extends should return unchanged."""
        from llm_eval.cli import _resolve_extends

        config = {"judge": {"model": "gpt-4o"}}
        result = _resolve_extends(config, "/tmp")
        assert result == config
