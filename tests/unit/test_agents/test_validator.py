"""Tests for ValidatorAgent (Self-RAG)."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.base import AgentContext, AgentStatus
from src.agents.validator.agent import ValidatorAgent
from src.llm.client import LLMClient, LLMResponse


@pytest.fixture
def mock_llm() -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.generate = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps({
                "retrieve_needed": False,
                "retrieve_reason": "충분한 증거",
                "relevance": {"score": 0.9, "irrelevant_sources": [], "explanation": "관련 있음"},
                "support": {
                    "level": "Fully",
                    "supported_claims": ["CPU 급증은 배치 작업으로 인한 것"],
                    "unsupported_claims": [],
                    "explanation": "증거로 뒷받침됨",
                },
                "usefulness": {"score": 4, "explanation": "유용한 답변"},
                "overall_confidence": 0.85,
                "hallucination_flags": [],
                "improvement_suggestions": [],
            }),
            provider="mock",
            model="mock",
            input_tokens=500,
            output_tokens=300,
            elapsed_ms=600.0,
        )
    )
    return client


class TestValidatorAgent:
    async def test_validation_passes(self, mock_llm: MagicMock) -> None:
        validator = ValidatorAgent(llm_client=mock_llm)
        context = AgentContext(
            query="CPU 급증 원인은?",
            previous_results={
                "answer": "배치 작업이 원인입니다.",
                "confidence": 0.8,
                "documents": [{"content": "CPU가 95%까지 올랐습니다", "source_type": "vector"}],
            },
        )

        result = await validator.run(context)

        assert result.status == AgentStatus.SUCCESS
        assert result.data["is_valid"] is True
        assert result.data["adjusted_confidence"] > 0
        assert result.data["support_level"] == "Fully"

    async def test_validation_fails_no_support(self, mock_llm: MagicMock) -> None:
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps({
                    "retrieve_needed": True,
                    "retrieve_reason": "증거 부족",
                    "relevance": {"score": 0.3, "irrelevant_sources": ["1"], "explanation": "관련 없음"},
                    "support": {"level": "No", "supported_claims": [], "unsupported_claims": ["전체"], "explanation": "뒷받침 안됨"},
                    "usefulness": {"score": 1, "explanation": "유용하지 않음"},
                    "overall_confidence": 0.1,
                    "hallucination_flags": ["답변이 증거와 무관"],
                    "improvement_suggestions": ["재검색 필요"],
                }),
                provider="mock",
                model="mock",
                input_tokens=500,
                output_tokens=300,
                elapsed_ms=600.0,
            )
        )

        validator = ValidatorAgent(llm_client=mock_llm)
        context = AgentContext(
            query="테스트",
            previous_results={
                "answer": "잘못된 답변",
                "confidence": 0.2,
                "documents": [],
            },
        )

        result = await validator.run(context)

        assert result.data["is_valid"] is False
        assert result.data["needs_retry"] is True
        assert len(result.data["hallucination_flags"]) > 0

    async def test_confidence_adjustment(self) -> None:
        adjusted = ValidatorAgent._adjust_confidence(0.8, "Fully", 5, 0)
        assert adjusted >= 0.8  # Fully supported + high usefulness (1.1 * 1.0, capped at 1.0)

        adjusted = ValidatorAgent._adjust_confidence(0.8, "No", 1, 2)
        assert adjusted < 0.3  # Not supported + low usefulness + hallucinations
