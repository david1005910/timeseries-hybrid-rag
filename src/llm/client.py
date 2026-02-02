"""LLM client with multi-provider support and retry logic."""
from __future__ import annotations

import time
from typing import Any, AsyncIterator

import anthropic
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LLMClient:
    """멀티 프로바이더 LLM 클라이언트 (Anthropic Claude + OpenAI fallback)."""

    def __init__(self) -> None:
        settings = get_settings()
        self._anthropic: anthropic.AsyncAnthropic | None = None
        self._openai: openai.AsyncOpenAI | None = None

        if settings.anthropic_api_key:
            self._anthropic = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        if settings.openai_api_key:
            self._openai = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=30))
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        model: str | None = None,
    ) -> LLMResponse:
        """LLM 응답 생성.

        Anthropic Claude를 우선 사용하고, 실패 시 OpenAI로 fallback.
        """
        t0 = time.time()

        # Try Anthropic first
        if self._anthropic:
            try:
                return await self._generate_anthropic(
                    prompt, system_prompt, max_tokens, temperature, model, t0
                )
            except Exception as e:
                logger.warning("anthropic_failed", error=str(e))

        # Fallback to OpenAI
        if self._openai:
            return await self._generate_openai(
                prompt, system_prompt, max_tokens, temperature, model, t0
            )

        raise RuntimeError("No LLM provider configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY.")

    async def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        model: str | None,
        t0: float,
    ) -> LLMResponse:
        assert self._anthropic is not None
        model_name = model or "claude-sonnet-4-20250514"
        messages = [{"role": "user", "content": prompt}]
        kwargs: dict[str, Any] = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self._anthropic.messages.create(**kwargs)
        content = response.content[0].text
        elapsed = (time.time() - t0) * 1000

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

        logger.info(
            "llm_generate",
            provider="anthropic",
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_ms=round(elapsed, 2),
        )

        return LLMResponse(
            content=content,
            provider="anthropic",
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_ms=elapsed,
        )

    async def _generate_openai(
        self,
        prompt: str,
        system_prompt: str | None,
        max_tokens: int,
        temperature: float,
        model: str | None,
        t0: float,
    ) -> LLMResponse:
        assert self._openai is not None
        model_name = model or "gpt-4o"
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self._openai.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content or ""
        elapsed = (time.time() - t0) * 1000

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        logger.info(
            "llm_generate",
            provider="openai",
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_ms=round(elapsed, 2),
        )

        return LLMResponse(
            content=content,
            provider="openai",
            model=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            elapsed_ms=elapsed,
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> AsyncIterator[str]:
        """스트리밍 응답 생성. Anthropic 실패 시 OpenAI로 fallback."""
        if self._anthropic:
            try:
                model_name = "claude-sonnet-4-20250514"
                messages = [{"role": "user", "content": prompt}]
                kwargs: dict[str, Any] = {
                    "model": model_name,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": messages,
                }
                if system_prompt:
                    kwargs["system"] = system_prompt

                async with self._anthropic.messages.stream(**kwargs) as stream:
                    async for text in stream.text_stream:
                        yield text
                return
            except Exception as e:
                logger.warning("anthropic_stream_failed", error=str(e))

        if self._openai:
            messages_list: list[dict[str, str]] = []
            if system_prompt:
                messages_list.append({"role": "system", "content": system_prompt})
            messages_list.append({"role": "user", "content": prompt})

            stream = await self._openai.chat.completions.create(
                model="gpt-4o",
                messages=messages_list,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content


class LLMResponse:
    """LLM 응답 결과."""

    def __init__(
        self,
        content: str,
        provider: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        elapsed_ms: float = 0.0,
    ) -> None:
        self.content = content
        self.provider = provider
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.elapsed_ms = elapsed_ms
        self.total_tokens = input_tokens + output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """비용 추정 (Claude Sonnet 기준)."""
        if self.provider == "anthropic":
            return (self.input_tokens * 3 + self.output_tokens * 15) / 1_000_000
        elif self.provider == "openai":
            return (self.input_tokens * 5 + self.output_tokens * 15) / 1_000_000
        return 0.0
