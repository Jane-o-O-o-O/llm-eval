"""Tests for the SQLite judge cache (llm_eval.cache)."""

from __future__ import annotations

import os

from llm_eval.cache import JudgeCache


class TestJudgeCache:
    """Tests for JudgeCache initialization and operations."""

    def test_init_creates_db(self, tmp_path):
        db = str(tmp_path / "test_cache.db")
        cache = JudgeCache(db_path=db)
        assert os.path.exists(db)
        cache.close()

    def test_get_miss_returns_none(self, tmp_path):
        cache = JudgeCache(db_path=str(tmp_path / "cache.db"))
        result = cache.get(model="gpt-4o", temperature=0, prompt="test prompt")
        assert result is None
        cache.close()

    def test_set_and_get(self, tmp_path):
        cache = JudgeCache(db_path=str(tmp_path / "cache.db"))
        response = {"score": 0.9, "reasoning": "Good answer"}
        cache.set(model="gpt-4o", temperature=0, prompt="test prompt", response=response)

        cached = cache.get(model="gpt-4o", temperature=0, prompt="test prompt")
        assert cached == response
        cache.close()

    def test_different_model_miss(self, tmp_path):
        cache = JudgeCache(db_path=str(tmp_path / "cache.db"))
        cache.set(model="gpt-4o", temperature=0, prompt="test", response={"score": 0.5})

        # Different model should miss
        result = cache.get(model="gpt-3.5-turbo", temperature=0, prompt="test")
        assert result is None
        cache.close()

    def test_different_temperature_miss(self, tmp_path):
        cache = JudgeCache(db_path=str(tmp_path / "cache.db"))
        cache.set(model="gpt-4o", temperature=0, prompt="test", response={"score": 0.5})

        result = cache.get(model="gpt-4o", temperature=0.5, prompt="test")
        assert result is None
        cache.close()

    def test_different_prompt_miss(self, tmp_path):
        cache = JudgeCache(db_path=str(tmp_path / "cache.db"))
        cache.set(model="gpt-4o", temperature=0, prompt="prompt A", response={"score": 0.5})

        result = cache.get(model="gpt-4o", temperature=0, prompt="prompt B")
        assert result is None
        cache.close()

    def test_overwrite(self, tmp_path):
        cache = JudgeCache(db_path=str(tmp_path / "cache.db"))
        cache.set(model="gpt-4o", temperature=0, prompt="test", response={"score": 0.5})
        cache.set(model="gpt-4o", temperature=0, prompt="test", response={"score": 0.9})

        cached = cache.get(model="gpt-4o", temperature=0, prompt="test")
        assert cached == {"score": 0.9}
        cache.close()

    def test_clear(self, tmp_path):
        cache = JudgeCache(db_path=str(tmp_path / "cache.db"))
        cache.set(model="gpt-4o", temperature=0, prompt="p1", response={"score": 0.5})
        cache.set(model="gpt-4o", temperature=0, prompt="p2", response={"score": 0.8})

        removed = cache.clear()
        assert removed == 2
        assert cache.get(model="gpt-4o", temperature=0, prompt="p1") is None
        cache.close()

    def test_stats(self, tmp_path):
        cache = JudgeCache(db_path=str(tmp_path / "cache.db"))
        cache.set(model="gpt-4o", temperature=0, prompt="test", response={"score": 0.5})

        stats = cache.stats()
        assert stats["entry_count"] == 1
        assert stats["db_path"] == str(tmp_path / "cache.db")
        assert stats["db_size_bytes"] > 0
        cache.close()

    def test_stats_empty(self, tmp_path):
        cache = JudgeCache(db_path=str(tmp_path / "cache.db"))
        stats = cache.stats()
        assert stats["entry_count"] == 0
        cache.close()

    def test_eviction(self, tmp_path):
        cache = JudgeCache(db_path=str(tmp_path / "cache.db"), max_entries=3)
        for i in range(5):
            cache.set(model="gpt-4o", temperature=0, prompt=f"prompt-{i}", response={"i": i})

        stats = cache.stats()
        assert stats["entry_count"] == 3
        # Oldest entries should be evicted
        assert cache.get(model="gpt-4o", temperature=0, prompt="prompt-0") is None
        assert cache.get(model="gpt-4o", temperature=0, prompt="prompt-4") is not None
        cache.close()

    def test_eviction_disabled(self, tmp_path):
        cache = JudgeCache(db_path=str(tmp_path / "cache.db"), max_entries=0)
        for i in range(5):
            cache.set(model="gpt-4o", temperature=0, prompt=f"p-{i}", response={})
        assert cache.stats()["entry_count"] == 5
        cache.close()

    def test_unicode_content(self, tmp_path):
        cache = JudgeCache(db_path=str(tmp_path / "cache.db"))
        prompt = "评价这个回答：你好世界"
        response = {"score": 0.8, "reasoning": "回答很好"}
        cache.set(model="gpt-4o", temperature=0, prompt=prompt, response=response)
        cached = cache.get(model="gpt-4o", temperature=0, prompt=prompt)
        assert cached == response
        cache.close()

    def test_complex_response(self, tmp_path):
        cache = JudgeCache(db_path=str(tmp_path / "cache.db"))
        response = {
            "score": 0.75,
            "reasoning": "Partially correct",
            "claims": ["claim1", "claim2"],
            "details": {"supported": 3, "total": 4},
        }
        cache.set(model="gpt-4o", temperature=0, prompt="test", response=response)
        cached = cache.get(model="gpt-4o", temperature=0, prompt="test")
        assert cached == response
        cache.close()

    def test_default_path(self):
        """Test that default path resolves to ~/.llm-eval/cache.db."""
        cache = JudgeCache.__new__(JudgeCache)
        cache._db_path = None
        cache._max_entries = 1000
        # Just verify the constant
        from llm_eval.cache import _DEFAULT_CACHE_DB
        assert ".llm-eval" in _DEFAULT_CACHE_DB
        assert "cache.db" in _DEFAULT_CACHE_DB
