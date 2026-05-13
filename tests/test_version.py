"""Tests for version consistency."""

from __future__ import annotations

from pathlib import Path

import tomllib


class TestVersionConsistency:
    """Ensure __version__ matches pyproject.toml."""

    def test_version_matches_pyproject(self) -> None:
        """Test that llm_eval.__version__ matches the version in pyproject.toml."""
        from llm_eval import __version__

        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        assert __version__ == config["project"]["version"], (
            f"Version mismatch: __version__={__version__} "
            f"vs pyproject.toml={config['project']['version']}"
        )

    def test_version_format(self) -> None:
        """Test that version follows semver format."""
        from llm_eval import __version__

        parts = __version__.split(".")
        assert len(parts) >= 2, f"Version should have at least major.minor: {__version__}"
        for part in parts:
            assert part.isdigit(), f"Version part should be numeric: {part}"
