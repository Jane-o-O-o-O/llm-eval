"""Tests for config schema command — export JSON Schema for YAML config."""

from __future__ import annotations

import json
import os

import pytest
from click.testing import CliRunner

from llm_eval.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestConfigSchema:
    """Test the config schema command."""

    def test_schema_output_json(self, runner):
        """config schema outputs valid JSON Schema."""
        result = runner.invoke(main, ["config", "schema"])
        assert result.exit_code == 0
        schema = json.loads(result.output)
        assert "$schema" in schema
        assert "properties" in schema

    def test_schema_has_judge_section(self, runner):
        """Schema includes judge configuration section."""
        result = runner.invoke(main, ["config", "schema"])
        schema = json.loads(result.output)
        assert "judge" in schema["properties"]

    def test_schema_has_evaluations(self, runner):
        """Schema includes evaluations section."""
        result = runner.invoke(main, ["config", "schema"])
        schema = json.loads(result.output)
        assert "evaluations" in schema["properties"]

    def test_schema_has_defaults(self, runner):
        """Schema includes defaults section."""
        result = runner.invoke(main, ["config", "schema"])
        schema = json.loads(result.output)
        assert "defaults" in schema["properties"]

    def test_schema_to_file(self, tmp_path, runner):
        """config schema --output writes to file."""
        output_path = str(tmp_path / "schema.json")
        result = runner.invoke(main, ["config", "schema", "-o", output_path])
        assert result.exit_code == 0
        assert os.path.exists(output_path)

        with open(output_path) as f:
            schema = json.load(f)
        assert "$schema" in schema

    def test_schema_judge_properties(self, runner):
        """Schema judge section has model, temperature, etc."""
        result = runner.invoke(main, ["config", "schema"])
        schema = json.loads(result.output)
        judge_props = schema["properties"]["judge"]["properties"]
        assert "model" in judge_props
        assert "temperature" in judge_props
        assert "base_url" in judge_props
