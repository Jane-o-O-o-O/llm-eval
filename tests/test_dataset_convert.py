"""Tests for dataset convert command."""

from __future__ import annotations

import json
import os
import tempfile

from click.testing import CliRunner

from llm_eval.cli import main


class TestDatasetConvert:
    """Test dataset convert between JSONL and CSV."""

    def _make_jsonl(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(json.dumps({"query": "q1", "context": ["c1"], "answer": "a1"}) + "\n")
            f.write(json.dumps({"query": "q2", "context": ["c2"], "answer": "a2", "reference": "r2"}) + "\n")

    def test_jsonl_to_csv(self):
        """Convert JSONL to CSV."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "input.jsonl")
            dst = os.path.join(tmpdir, "output.csv")
            self._make_jsonl(src)
            result = runner.invoke(main, ["dataset", "convert", src, dst])
            assert result.exit_code == 0
            assert os.path.exists(dst)
            with open(dst) as f:
                content = f.read()
            assert "query" in content
            assert "q1" in content

    def test_csv_to_jsonl(self):
        """Convert CSV to JSONL."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "input.csv")
            jsonl_path = os.path.join(tmpdir, "output.jsonl")
            with open(csv_path, "w") as f:
                f.write("query,context,answer\n")
                f.write("q1,c1,a1\n")
                f.write("q2,c2,a2\n")
            result = runner.invoke(main, ["dataset", "convert", csv_path, jsonl_path])
            assert result.exit_code == 0
            assert os.path.exists(jsonl_path)
            with open(jsonl_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            assert len(lines) == 2
            data = json.loads(lines[0])
            assert data["query"] == "q1"

    def test_convert_help(self):
        """convert --help should show usage."""
        runner = CliRunner()
        result = runner.invoke(main, ["dataset", "convert", "--help"])
        assert result.exit_code == 0
        assert "SOURCE" in result.output or "source" in result.output.lower()
