"""Tests for Judge cache integration, auth headers, and new features."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_eval.judge import Judge
from llm_eval.models import JudgeConfig


class TestJudgeCacheIntegration:
    """Test that Judge properly integrates with JudgeCache."""

    @pytest.fixture
    def cache_db(self, tmp_path):
        """Create a temporary cache database."""
        db_path = str(tmp_path / "test_cache.db")
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
        return db_path

    @pytest.fixture
    def cache(self, cache_db):
        """Create a JudgeCache instance with temporary DB."""
        from llm_eval.cache import JudgeCache
        return JudgeCache(db_path=cache_db)

    def test_judge_accepts_cache_param(self):
        """Judge should accept cache and use_cache parameters."""
        judge = Judge(config=JudgeConfig(), cache=None, use_cache=True)
        assert judge.cache is None
        assert judge.use_cache is True

    def test_judge_default_no_cache(self):
        """Judge should default to no cache."""
        judge = Judge()
        assert judge.cache is None
        assert judge.use_cache is True

    def test_judge_use_cache_false(self):
        """Judge with use_cache=False should not use cache."""
        judge = Judge(cache=MagicMock(), use_cache=False)
        assert judge.use_cache is False

    @pytest.mark.asyncio
    async def test_judge_returns_cached_response(self, cache):
        """Judge should return cached response when available."""
        # Pre-populate cache
        prompt = "Test prompt"
        cached_response = {"score": 0.95, "reasoning": "cached"}
        cache.set(
            model="gpt-4o",
            temperature=0.0,
            prompt=prompt,
            response=cached_response,
        )

        judge = Judge(config=JudgeConfig(), cache=cache, use_cache=True)
        result = await judge.call(prompt)

        assert result == cached_response

    @pytest.mark.asyncio
    async def test_judge_stores_response_in_cache(self, cache):
        """Judge should store API response in cache."""
        judge = Judge(config=JudgeConfig(), cache=cache, use_cache=True)

        mock_response = {"score": 0.8, "reasoning": "good"}
        with patch.object(judge, "_build_headers", return_value={}):
            with patch("httpx.AsyncClient") as mock_client:
                # Setup mock HTTP response
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.raise_for_status = MagicMock()
                mock_resp.json.return_value = {
                    "choices": [{"message": {"content": json.dumps(mock_response)}}]
                }
                mock_instance = AsyncMock()
                mock_instance.post.return_value = mock_resp
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                mock_client.return_value = mock_instance

                result = await judge.call("New prompt")

        assert result == mock_response

        # Verify it was cached
        cached = cache.get(model="gpt-4o", temperature=0.0, prompt="New prompt")
        assert cached == mock_response

    @pytest.mark.asyncio
    async def test_judge_no_cache_bypasses(self, cache):
        """Judge with use_cache=False should bypass cache entirely."""
        # Pre-populate cache
        cache.set(
            model="gpt-4o",
            temperature=0.0,
            prompt="test",
            response={"score": 0.99},
        )

        judge = Judge(config=JudgeConfig(), cache=cache, use_cache=False)

        mock_response = {"score": 0.5, "reasoning": "different"}
        with patch.object(judge, "_build_headers", return_value={}):
            with patch("httpx.AsyncClient") as mock_client:
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.raise_for_status = MagicMock()
                mock_resp.json.return_value = {
                    "choices": [{"message": {"content": json.dumps(mock_response)}}]
                }
                mock_instance = AsyncMock()
                mock_instance.post.return_value = mock_resp
                mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
                mock_instance.__aexit__ = AsyncMock(return_value=False)
                mock_client.return_value = mock_instance

                result = await judge.call("test")

        # Should get the API response, not the cached one
        assert result == mock_response


class TestJudgeAuthHeaders:
    """Test that Judge builds correct auth headers."""

    def test_auth_header_with_openai_key(self, monkeypatch):
        """Should include Authorization header with OPENAI_API_KEY."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-12345")
        judge = Judge(config=JudgeConfig())
        headers = judge._build_headers()
        assert headers["Authorization"] == "Bearer sk-test-12345"

    def test_auth_header_with_explicit_key(self):
        """Should use explicit api_key over environment."""
        config = JudgeConfig(api_key="explicit-key")
        judge = Judge(config=config)
        headers = judge._build_headers()
        assert headers["Authorization"] == "Bearer explicit-key"

    def test_auth_header_anthropic_key(self, monkeypatch):
        """Should use ANTHROPIC_API_KEY for Anthropic base URLs."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-key-123")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = JudgeConfig(base_url="https://api.anthropic.com/v1")
        judge = Judge(config=config)
        headers = judge._build_headers()
        assert headers["Authorization"] == "Bearer ant-key-123"

    def test_auth_header_google_key(self, monkeypatch):
        """Should use GOOGLE_API_KEY for Google/Gemini base URLs."""
        monkeypatch.setenv("GOOGLE_API_KEY", "goog-key-123")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = JudgeConfig(base_url="https://generativelanguage.googleapis.com/v1")
        judge = Judge(config=config)
        headers = judge._build_headers()
        assert headers["Authorization"] == "Bearer goog-key-123"

    def test_no_auth_header_when_no_key(self, monkeypatch):
        """Should not include Authorization header when no key is available."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        judge = Judge(config=JudgeConfig())
        headers = judge._build_headers()
        assert "Authorization" not in headers

    def test_content_type_always_json(self):
        """Should always include Content-Type: application/json."""
        judge = Judge(config=JudgeConfig())
        headers = judge._build_headers()
        assert headers["Content-Type"] == "application/json"

    def test_get_api_key_explicit_over_env(self, monkeypatch):
        """Explicit api_key should take precedence over env vars."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        config = JudgeConfig(api_key="config-key")
        judge = Judge(config=config)
        assert judge._get_api_key() == "config-key"


class TestJudgeConfigApiKey:
    """Test that JudgeConfig properly handles api_key."""

    def test_judge_config_default_api_key(self):
        """JudgeConfig should default to None api_key."""
        config = JudgeConfig()
        assert config.api_key is None

    def test_judge_config_with_api_key(self):
        """JudgeConfig should accept explicit api_key."""
        config = JudgeConfig(api_key="test-key")
        assert config.api_key == "test-key"

    def test_eval_config_parses_api_key(self):
        """EvalConfig.from_dict should parse api_key from judge section."""
        from llm_eval.models import EvalConfig

        data = {
            "judge": {"model": "gpt-4o", "api_key": "yaml-key"},
            "evaluations": [],
        }
        config = EvalConfig.from_dict(data)
        assert config.judge.api_key == "yaml-key"

    def test_eval_config_no_api_key(self):
        """EvalConfig.from_dict should default api_key to None."""
        from llm_eval.models import EvalConfig

        data = {"judge": {"model": "gpt-4o"}, "evaluations": []}
        config = EvalConfig.from_dict(data)
        assert config.judge.api_key is None
