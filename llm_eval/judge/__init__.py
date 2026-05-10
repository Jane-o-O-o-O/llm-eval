"""LLM Judge adapter for making evaluation calls to language models."""

from __future__ import annotations

import asyncio
import json
import re

import httpx

from llm_eval.models import JudgeConfig


class Judge:
    """Adapter for calling LLM judge models via OpenAI-compatible API.

    Supports any model provider with an OpenAI-compatible chat completions endpoint.

    Attributes:
        config: The judge configuration.
    """

    def __init__(self, config: JudgeConfig | None = None) -> None:
        """Initialize the judge.

        Args:
            config: Judge configuration. Uses defaults if not provided.
        """
        self.config = config or JudgeConfig()

    @property
    def _base_url(self) -> str:
        """Get the API base URL."""
        return self.config.base_url or "https://api.openai.com/v1"

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
        messages = [
            {"role": "system", "content": "You are an evaluation judge. Respond only as instructed."},
            {"role": "user", "content": prompt},
        ]

        body: dict = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                    response = await client.post(
                        f"{self._base_url}/chat/completions",
                        json=body,
                        headers={"Content-Type": "application/json"},
                    )
                    response.raise_for_status()
                    data = response.json()
                    content = data["choices"][0]["message"]["content"]
                    if response_format == "json":
                        return self.parse_json_response(content)
                    return {"content": content}
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)

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
