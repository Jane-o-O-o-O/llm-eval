"""Shared test fixtures."""

from __future__ import annotations

import pytest
from llm_eval.models import Sample


@pytest.fixture
def sample() -> Sample:
    """A basic evaluation sample."""
    return Sample(
        query="What is Python?",
        context=["Python is a programming language."],
        answer="Python is a programming language.",
    )


@pytest.fixture
def sample_with_reference() -> Sample:
    """A sample with a reference answer."""
    return Sample(
        query="What is Python?",
        context=["Python is a programming language."],
        answer="Python is a programming language.",
        reference="Python is a widely-used programming language.",
    )


@pytest.fixture
def multiple_samples() -> list[Sample]:
    """Multiple samples for batch evaluation tests."""
    return [
        Sample(
            query=f"Question {i}",
            context=[f"Context {i}"],
            answer=f"Answer {i}",
        )
        for i in range(5)
    ]
