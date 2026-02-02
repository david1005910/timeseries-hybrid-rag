"""Tests for LLMClient multi-provider support and LLMResponse properties."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import RetryError

from src.llm.client import LLMClient, LLMResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_anthropic_response() -> MagicMock:
    """Anthropic messages.create response mock."""
    content_block = MagicMock()
    content_block.text = "Anthropic 응답 텍스트"

    usage = MagicMock()
    usage.input_tokens = 120
    usage.output_tokens = 80

    response = MagicMock()
    response.content = [content_block]
    response.usage = usage
    return response


@pytest.fixture
def mock_openai_response() -> MagicMock:
    """OpenAI chat.completions.create response mock."""
    message = MagicMock()
    message.content = "OpenAI 응답 텍스트"

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = 150
    usage.completion_tokens = 100

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_settings(**overrides: Any) -> MagicMock:
    """Create a mock Settings object with sensible defaults."""
    defaults = {
        "anthropic_api_key": "",
        "openai_api_key": "",
        "embedding_model": "text-embedding-3-large",
        "embedding_dimension": 1536,
    }
    defaults.update(overrides)
    settings = MagicMock()
    for key, value in defaults.items():
        setattr(settings, key, value)
    return settings


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLLMClient:
    """LLMClient.generate with Anthropic / OpenAI / fallback scenarios."""

    @patch("src.llm.client.get_settings")
    @patch("src.llm.client.anthropic.AsyncAnthropic")
    async def test_generate_anthropic_success(
        self,
        mock_anthropic_cls: MagicMock,
        mock_get_settings: MagicMock,
        mock_anthropic_response: MagicMock,
    ) -> None:
        """generate() should return an LLMResponse from Anthropic when configured."""
        mock_get_settings.return_value = _make_settings(anthropic_api_key="sk-ant-test")

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create = AsyncMock(return_value=mock_anthropic_response)
        mock_anthropic_cls.return_value = mock_client_instance

        client = LLMClient()
        result = await client.generate(prompt="테스트 질문", system_prompt="시스템 프롬프트")

        assert isinstance(result, LLMResponse)
        assert result.content == "Anthropic 응답 텍스트"
        assert result.provider == "anthropic"
        assert result.input_tokens == 120
        assert result.output_tokens == 80
        mock_client_instance.messages.create.assert_awaited_once()

    @patch("src.llm.client.get_settings")
    @patch("src.llm.client.openai.AsyncOpenAI")
    @patch("src.llm.client.anthropic.AsyncAnthropic")
    async def test_generate_falls_back_to_openai(
        self,
        mock_anthropic_cls: MagicMock,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
        mock_openai_response: MagicMock,
    ) -> None:
        """When Anthropic raises an exception, generate() should fall back to OpenAI."""
        mock_get_settings.return_value = _make_settings(
            anthropic_api_key="sk-ant-test",
            openai_api_key="sk-oai-test",
        )

        # Anthropic fails
        mock_ant_instance = MagicMock()
        mock_ant_instance.messages.create = AsyncMock(
            side_effect=Exception("Anthropic service error")
        )
        mock_anthropic_cls.return_value = mock_ant_instance

        # OpenAI succeeds
        mock_oai_instance = MagicMock()
        mock_oai_instance.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_openai_cls.return_value = mock_oai_instance

        client = LLMClient()
        result = await client.generate(prompt="폴백 테스트")

        assert result.provider == "openai"
        assert result.content == "OpenAI 응답 텍스트"
        assert result.input_tokens == 150
        assert result.output_tokens == 100
        mock_oai_instance.chat.completions.create.assert_awaited_once()

    @patch("src.llm.client.get_settings")
    async def test_generate_raises_when_no_provider(
        self, mock_get_settings: MagicMock
    ) -> None:
        """generate() should raise RetryError (wrapping RuntimeError) when no provider configured.

        The @retry decorator retries 3 times and then wraps the final
        RuntimeError in a tenacity.RetryError.
        """
        mock_get_settings.return_value = _make_settings(
            anthropic_api_key="", openai_api_key=""
        )

        client = LLMClient()
        with pytest.raises(RetryError) as exc_info:
            await client.generate(prompt="에러 테스트")

        # The underlying cause should be our RuntimeError
        last_attempt = exc_info.value.last_attempt
        assert last_attempt.failed
        underlying = last_attempt.exception()
        assert isinstance(underlying, RuntimeError)
        assert "No LLM provider configured" in str(underlying)

    @patch("src.llm.client.get_settings")
    @patch("src.llm.client.openai.AsyncOpenAI")
    async def test_generate_openai_with_system_prompt(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
        mock_openai_response: MagicMock,
    ) -> None:
        """OpenAI path should prepend system message when system_prompt is given."""
        mock_get_settings.return_value = _make_settings(openai_api_key="sk-oai-test")

        mock_oai_instance = MagicMock()
        mock_oai_instance.chat.completions.create = AsyncMock(return_value=mock_openai_response)
        mock_openai_cls.return_value = mock_oai_instance

        client = LLMClient()
        await client.generate(prompt="질문", system_prompt="시스템 지시")

        call_kwargs = mock_oai_instance.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "시스템 지시"
        assert messages[1]["role"] == "user"

    @patch("src.llm.client.get_settings")
    @patch("src.llm.client.anthropic.AsyncAnthropic")
    async def test_generate_stream_anthropic(
        self,
        mock_anthropic_cls: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """generate_stream() should yield text chunks from Anthropic streaming."""
        mock_get_settings.return_value = _make_settings(anthropic_api_key="sk-ant-test")

        chunks = ["첫 번째 ", "두 번째 ", "세 번째"]

        # Build an async iterator for text_stream
        async def fake_text_stream():
            for chunk in chunks:
                yield chunk

        # Build async context manager for messages.stream(...)
        stream_cm = MagicMock()
        stream_cm.text_stream = fake_text_stream()
        stream_cm.__aenter__ = AsyncMock(return_value=stream_cm)
        stream_cm.__aexit__ = AsyncMock(return_value=False)

        mock_ant_instance = MagicMock()
        mock_ant_instance.messages.stream = MagicMock(return_value=stream_cm)
        mock_anthropic_cls.return_value = mock_ant_instance

        client = LLMClient()
        collected: list[str] = []
        async for text in client.generate_stream(prompt="스트림 테스트"):
            collected.append(text)

        assert collected == chunks

    @patch("src.llm.client.get_settings")
    @patch("src.llm.client.openai.AsyncOpenAI")
    async def test_generate_stream_openai_fallback(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """generate_stream() should yield text chunks via OpenAI when Anthropic not configured."""
        mock_get_settings.return_value = _make_settings(openai_api_key="sk-oai-test")

        chunks_text = ["안녕", "하세", "요"]

        # Build OpenAI streaming response
        async def fake_stream():
            for text in chunks_text:
                chunk = MagicMock()
                delta = MagicMock()
                delta.content = text
                choice = MagicMock()
                choice.delta = delta
                chunk.choices = [choice]
                yield chunk

        mock_oai_instance = MagicMock()
        mock_oai_instance.chat.completions.create = AsyncMock(return_value=fake_stream())
        mock_openai_cls.return_value = mock_oai_instance

        client = LLMClient()
        collected: list[str] = []
        async for text in client.generate_stream(prompt="OpenAI 스트림"):
            collected.append(text)

        assert collected == chunks_text


class TestLLMResponse:
    """LLMResponse data class properties."""

    def test_total_tokens(self) -> None:
        """total_tokens should be the sum of input and output tokens."""
        resp = LLMResponse(
            content="test",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            input_tokens=200,
            output_tokens=100,
        )
        assert resp.total_tokens == 300

    def test_estimated_cost_anthropic(self) -> None:
        """estimated_cost_usd for anthropic: (input*3 + output*15) / 1_000_000."""
        resp = LLMResponse(
            content="cost test",
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            input_tokens=1000,
            output_tokens=500,
        )
        expected = (1000 * 3 + 500 * 15) / 1_000_000
        assert resp.estimated_cost_usd == pytest.approx(expected)

    def test_estimated_cost_openai(self) -> None:
        """estimated_cost_usd for openai: (input*5 + output*15) / 1_000_000."""
        resp = LLMResponse(
            content="cost test",
            provider="openai",
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
        )
        expected = (1000 * 5 + 500 * 15) / 1_000_000
        assert resp.estimated_cost_usd == pytest.approx(expected)

    def test_estimated_cost_unknown_provider(self) -> None:
        """Unknown providers should return 0.0 cost."""
        resp = LLMResponse(content="x", provider="unknown", model="m", input_tokens=100, output_tokens=50)
        assert resp.estimated_cost_usd == 0.0

    def test_default_values(self) -> None:
        """Token and elapsed defaults should be zero."""
        resp = LLMResponse(content="defaults", provider="anthropic", model="m")
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.elapsed_ms == 0.0
        assert resp.total_tokens == 0
