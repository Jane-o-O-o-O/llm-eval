"""Tests for sync SDK wrappers and the evaluate CLI command."""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner

from llm_eval.cli import main


@pytest.fixture
def runner():
    return CliRunner()


class TestSyncSDK:
    """Test the synchronous SDK wrappers."""

    def test_evaluate_sync_exists(self):
        """evaluate_sync is importable."""
        from llm_eval.sdk import evaluate_sync
        assert callable(evaluate_sync)

    def test_evaluate_file_sync_exists(self):
        """evaluate_file_sync is importable."""
        from llm_eval.sdk import evaluate_file_sync
        assert callable(evaluate_file_sync)

    def test_sync_exports_in_all(self):
        """Sync functions are in __all__."""
        from llm_eval.sdk import __all__
        assert "evaluate_sync" in __all__
        assert "evaluate_file_sync" in __all__

    def test_sync_wrappers_in_package_init(self):
        """Sync wrappers are exported from llm_eval package."""
        import llm_eval
        assert hasattr(llm_eval, "evaluate_sync")
        assert hasattr(llm_eval, "evaluate_file_sync")
