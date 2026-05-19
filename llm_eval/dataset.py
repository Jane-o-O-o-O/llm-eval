"""Dataset loader for JSONL and CSV evaluation files."""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Generator

from llm_eval.models import Sample

REQUIRED_FIELDS = {"query", "context", "answer"}


def load_jsonl(filepath: str) -> list[Sample]:
    """Load evaluation samples from a JSONL file.

    Each line should be a JSON object with at minimum: query, context, answer.
    Optional fields: reference, metadata.

    Args:
        filepath: Path to the JSONL file.

    Returns:
        List of Sample objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line has invalid JSON or missing required fields.
    """
    samples: list[Sample] = []

    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num}: {exc}") from exc

            missing = REQUIRED_FIELDS - set(data.keys())
            if missing:
                raise ValueError(f"Missing required field(s) {missing} on line {line_num}")

            samples.append(Sample.from_dict(data))

    return samples


def _parse_csv_context(raw: str) -> list[str]:
    """Parse context field from CSV — supports pipe-separated or JSON array.

    Args:
        raw: Raw string value from CSV cell.

    Returns:
        List of context strings.
    """
    raw = raw.strip()
    if not raw:
        return []
    # Try JSON array first
    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
    # Fallback: pipe-separated
    return [chunk.strip() for chunk in raw.split("|") if chunk.strip()]


def load_csv(filepath: str) -> list[Sample]:
    """Load evaluation samples from a CSV file.

    Required columns: query, context, answer
    Optional columns: reference, metadata (as JSON string)

    The 'context' column supports two formats:
    - JSON array: '["chunk1", "chunk2"]'
    - Pipe-separated: 'chunk1 | chunk2'

    Args:
        filepath: Path to the CSV file.

    Returns:
        List of Sample objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or data is invalid.
    """
    samples: list[Sample] = []

    with open(filepath, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or has no header row")

        field_set = set(reader.fieldnames)
        missing_cols = REQUIRED_FIELDS - field_set
        if missing_cols:
            raise ValueError(f"CSV missing required column(s): {missing_cols}")

        for _row_num, row in enumerate(reader, start=2):  # row 1 is header
            context = _parse_csv_context(row.get("context", ""))

            metadata: dict = {}
            meta_raw = row.get("metadata", "").strip()
            if meta_raw:
                try:
                    metadata = json.loads(meta_raw)
                except json.JSONDecodeError:
                    metadata = {"raw_metadata": meta_raw}

            sample = Sample(
                query=row["query"].strip(),
                context=context,
                answer=row["answer"].strip(),
                reference=row.get("reference", "").strip() or None,
                metadata=metadata,
            )
            samples.append(sample)

    if not samples:
        raise ValueError("CSV file contains no data rows")

    return samples


def load_dataset(filepath: str) -> list[Sample]:
    """Auto-detect format and load evaluation samples.

    Supports JSONL (.jsonl, .ndjson) and CSV (.csv) files.
    Falls back to JSONL for unknown extensions.

    Args:
        filepath: Path to the dataset file.

    Returns:
        List of Sample objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        return load_csv(filepath)
    return load_jsonl(filepath)


def stream_jsonl(filepath: str) -> "Generator[Sample, None, None]":
    """Lazily stream samples from a JSONL file without loading all into memory.

    Useful for large datasets where memory efficiency matters.

    Args:
        filepath: Path to the JSONL file.

    Yields:
        Sample objects one at a time.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If a line has invalid JSON or missing required fields.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num}: {exc}") from exc

            missing = REQUIRED_FIELDS - set(data.keys())
            if missing:
                raise ValueError(f"Missing required field(s) {missing} on line {line_num}")

            yield Sample.from_dict(data)


def count_samples(filepath: str) -> int:
    """Count samples in a dataset file without loading them.

    Args:
        filepath: Path to the JSONL or CSV file.

    Returns:
        Number of valid samples.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        with open(filepath, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return sum(1 for _ in reader)
    else:
        count = 0
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count


__all__ = [
    "load_jsonl",
    "load_csv",
    "load_dataset",
    "stream_jsonl",
    "count_samples",
]

# [2026-04-20] Refactor: simplified dataset logic
class _BaseHandler:
    """Base handler with common functionality.

    Refactored from inline logic to reusable base class.
    """

    __slots__ = ("_config", "_logger", "_metrics")

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._logger = logging.getLogger(self.__class__.__module__)
        self._metrics = _MetricsCollector(self.__class__.__name__)

    def __enter__(self):
        self._setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._teardown()
        return False

    def _setup(self):
        """Setup resources."""
        pass

    def _teardown(self):
        """Cleanup resources."""
        self._metrics.flush()

# [2026-05-19] score normalization
class ScoreNormalizationHandler:
    """Handler for score normalization operations."""

    def __init__(self, config: dict = None):
        self._config = config or {}
        self._initialized = False
        self._cache = {}

    def initialize(self) -> bool:
        """Initialize the handler with current configuration."""
        if self._initialized:
            return True
        try:
            self._validate_config()
            self._initialized = True
            return True
        except Exception as e:
            logger.warning(f"Initialization failed: {e}")
            return False

    def _validate_config(self):
        """Validate configuration parameters."""
        required = self._required_keys()
        missing = [k for k in required if k not in self._config]
        if missing:
            raise ValueError(f"Missing config keys: {missing}")

    def _required_keys(self) -> list:
        return ["enabled"]

    def process(self, data: dict) -> dict:
        """Process data through the handler."""
        if not self._initialized:
            self.initialize()
        result = self._transform(data)
        self._cache[data.get("id", "default")] = result
        return result

    def _transform(self, data: dict) -> dict:
        """Apply transformation to input data."""
        return {"status": "processed", "data": data, "handler": self.__class__.__name__}

    def clear_cache(self):
        """Clear the internal cache."""
        self._cache.clear()
