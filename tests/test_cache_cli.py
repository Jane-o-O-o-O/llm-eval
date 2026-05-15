"""Tests for cache management CLI commands (stats, clear, purge)."""

from __future__ import annotations

import json
import os
import sqlite3

import pytest
from click.testing import CliRunner

from llm_eval.cli import main


@pytest.fixture
def cache_db(tmp_path):
    """Create a temporary cache database with some entries."""
    db_path = str(tmp_path / "test_cache.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS judge_cache (
            cache_key TEXT PRIMARY KEY,
            response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )
    for i in range(5):
        conn.execute(
            "INSERT INTO judge_cache (cache_key, response) VALUES (?, ?)",
            (f"key_{i}", json.dumps({"score": 0.5 + i * 0.1})),
        )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def runner():
    return CliRunner()


class TestCacheStats:
    """Test the cache stats command."""

    def test_cache_stats_shows_entries(self, cache_db, monkeypatch, runner):
        """Cache stats shows entry count."""
        monkeypatch.setattr("llm_eval.cache._DEFAULT_CACHE_DB", cache_db)
        result = runner.invoke(main, ["cache", "stats"])
        assert result.exit_code == 0
        assert "5" in result.output

    def test_cache_stats_json_output(self, cache_db, monkeypatch, runner):
        """Cache stats supports --output json."""
        monkeypatch.setattr("llm_eval.cache._DEFAULT_CACHE_DB", cache_db)
        result = runner.invoke(main, ["cache", "stats", "--output", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["entry_count"] == 5

    def test_cache_stats_empty(self, tmp_path, monkeypatch, runner):
        """Cache stats works with empty cache."""
        db_path = str(tmp_path / "empty_cache.db")
        conn = sqlite3.connect(db_path)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS judge_cache (
                cache_key TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )"""
        )
        conn.commit()
        conn.close()
        monkeypatch.setattr("llm_eval.cache._DEFAULT_CACHE_DB", db_path)
        result = runner.invoke(main, ["cache", "stats"])
        assert result.exit_code == 0
        assert "0" in result.output


class TestCacheClear:
    """Test the cache clear command."""

    def test_cache_clear_with_yes(self, cache_db, monkeypatch, runner):
        """Cache clear with --yes removes all entries."""
        monkeypatch.setattr("llm_eval.cache._DEFAULT_CACHE_DB", cache_db)
        result = runner.invoke(main, ["cache", "clear", "--yes"])
        assert result.exit_code == 0
        assert "5" in result.output

        # Verify cache is empty
        conn = sqlite3.connect(cache_db)
        count = conn.execute("SELECT COUNT(*) FROM judge_cache").fetchone()[0]
        conn.close()
        assert count == 0

    def test_cache_clear_no_flag(self, cache_db, monkeypatch, runner):
        """Cache clear without --yes should not delete."""
        monkeypatch.setattr("llm_eval.cache._DEFAULT_CACHE_DB", cache_db)
        result = runner.invoke(main, ["cache", "clear"], input="n\n")
        # Either prompts or requires --yes
        conn = sqlite3.connect(cache_db)
        count = conn.execute("SELECT COUNT(*) FROM judge_cache").fetchone()[0]
        conn.close()
        assert count == 5  # Unchanged


class TestCachePurge:
    """Test the cache purge command."""

    def test_cache_purge_older_than(self, cache_db, monkeypatch, runner):
        """Cache purge --older-than removes old entries."""
        monkeypatch.setattr("llm_eval.cache._DEFAULT_CACHE_DB", cache_db)
        # All entries are just created, so purge with 0 days = all removed
        result = runner.invoke(main, ["cache", "purge", "--older-than", "0", "--yes"])
        assert result.exit_code == 0
