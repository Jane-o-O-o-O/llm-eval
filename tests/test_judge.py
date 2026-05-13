"""Tests for the LLM judge adapter."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_eval.judge import Judge
from llm_eval.models import JudgeConfig


class TestJudge:
    """Tests for the Judge LLM adapter."""

    def test_create_judge_with_config(self) -> None:
        config = JudgeConfig(model="gpt-4o", temperature=0.1)
        judge = Judge(config)
        assert judge.config.model == "gpt-4o"
        assert judge.config.temperature == 0.1

    def test_create_judge_default_config(self) -> None:
        judge = Judge()
        assert judge.config.model == "gpt-4o"
        assert judge.config.temperature == 0.0

    @pytest.mark.asyncio
    async def test_judge_call_success(self) -> None:
        judge = Judge()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"score": 0.9, "reasoning": "Good answer"}'}}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
            result = await judge.call("What is 2+2?", response_format="json")
            assert result["score"] == 0.9
            assert result["reasoning"] == "Good answer"

    @pytest.mark.asyncio
    async def test_judge_call_retry_on_failure(self) -> None:
        import httpx

        judge = Judge(JudgeConfig(max_retries=2))

        error_response = MagicMock()
        error_response.status_code = 429
        error_response.text = "Rate limited"

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = [
                httpx.HTTPStatusError("429", request=MagicMock(), response=error_response),
                MagicMock(
                    status_code=200,
                    json=lambda: {"choices": [{"message": {"content": '{"score": 1.0}'}}]},
                    raise_for_status=MagicMock(),
                ),
            ]
            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await judge.call("test prompt")
                assert result["score"] == 1.0

    @pytest.mark.asyncio
    async def test_judge_parse_json_from_text(self) -> None:
        judge = Judge()
        text = 'Here is the result: {"score": 0.75, "reasoning": "Partial match"}'
        result = judge.parse_json_response(text)
        assert result["score"] == 0.75
        assert result["reasoning"] == "Partial match"

    @pytest.mark.asyncio
    async def test_judge_parse_json_no_json_raises(self) -> None:
        judge = Judge()
        with pytest.raises(ValueError, match="No JSON"):
            judge.parse_json_response("This is just plain text with no JSON.")
