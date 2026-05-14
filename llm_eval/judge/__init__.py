"""LLM Judge adapter for making evaluation calls to language models."""

from __future__ import annotations

import asyncio
import json
import os
import re

import httpx

from llm_eval.models import JudgeConfig


class Judge:
    """Adapter for calling LLM judge models via OpenAI-compatible API.

    Supports any model provider with an OpenAI-compatible chat completions endpoint.
    Uses response caching when a JudgeCache is provided to avoid redundant API calls.

    Attributes:
        config: The judge configuration.
        cache: Optional response cache instance.
        use_cache: Whether to use caching (can be disabled per-call).
    """

    def __init__(
        self,
        config: JudgeConfig | None = None,
        cache: "JudgeCache | None" = None,
        use_cache: bool = True,
    ) -> None:
        """Initialize the judge.

        Args:
            config: Judge configuration. Uses defaults if not provided.
            cache: Optional JudgeCache instance for caching responses.
            use_cache: Whether to use the cache. Set to False to bypass.
        """
        self.config = config or JudgeConfig()
        self.cache = cache
        self.use_cache = use_cache

    @property
    def _base_url(self) -> str:
        """Get the API base URL."""
        return self.config.base_url or "https://api.openai.com/v1"

    def _get_api_key(self) -> str | None:
        """Resolve API key from config or environment.

        Checks config.api_key first, then falls back to environment variables:
        - OPENAI_API_KEY (default)
        - ANTHROPIC_API_KEY (for Anthropic models)
        - Custom provider-specific keys based on base_url

        Returns:
            API key string or None if not found.
        """
        if self.config.api_key:
            return self.config.api_key

        # Check environment based on provider
        base_url = (self.config.base_url or "").lower()
        if "anthropic" in base_url:
            return os.environ.get("ANTHROPIC_API_KEY")
        if "google" in base_url or "gemini" in base_url:
            return os.environ.get("GOOGLE_API_KEY")
        return os.environ.get("OPENAI_API_KEY")

    def _build_headers(self) -> dict[str, str]:
        """Build HTTP headers including authentication.

        Returns:
            Dictionary of HTTP headers.
        """
        headers: dict[str, str] = {"Content-Type": "application/json"}
        api_key = self._get_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    async def call(self, prompt: str, response_format: str = "json") -> dict:
        """Call the judge model with a prompt and return parsed response.

        Args:
            prompt: The prompt to send to the judge.
            response_format: Expected response format ('json' or 'text').

        Returns:
            Parsed response dictionary.

        Raises:
            httpx.HTTPStatusError: After all retries are exhausted.
            ValueError: If JSON parsing fails for json format.
        """
        # Check cache first
        if self.use_cache and self.cache is not None:
            cached = self.cache.get(
                model=self.config.model,
                temperature=self.config.temperature,
                prompt=prompt,
            )
            if cached is not None:
                return cached

        messages = [
            {
                "role": "system",
                "content": "You are an evaluation judge. Respond only as instructed.",
            },
            {"role": "user", "content": prompt},
        ]

        body: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        headers = self._build_headers()
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                    response = await client.post(
                        f"{self._base_url}/chat/completions",
                        json=body,
                        headers=headers,
                    )
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    if response_format == "json":
                        result = self.parse_json_response(content)
                    else:
                        result = {"content": content}

                    # Store in cache
                    if self.use_cache and self.cache is not None:
                        self.cache.set(
                            model=self.config.model,
                            temperature=self.config.temperature,
                            prompt=prompt,
                            response=result,
                        )
                    return result
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2**attempt)

        raise last_error  # type: ignore[misc]

    @staticmethod
    def parse_json_response(text: str) -> dict:
        """Extract and parse JSON from a text response.

        Handles cases where the LLM wraps JSON in markdown code blocks
        or includes additional text before/after the JSON.

        Args:
            text: Raw text response from the LLM.

        Returns:
            Parsed JSON as a dictionary.

        Raises:
            ValueError: If no valid JSON can be extracted.
        """
        # Try parsing the entire text as JSON first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code blocks
        code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if code_block:
            try:
                return json.loads(code_block.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding a JSON object in the text
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"No JSON found in response: {text[:200]}")


__all__ = ["Judge"]
