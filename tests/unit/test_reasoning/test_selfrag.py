"""Tests for Self-RAG Verifier."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.llm.client import LLMClient, LLMResponse
from src.reasoning.selfrag.tokens import (
    RelevanceDecision,
    RelevanceToken,
    RetrieveDecision,
    RetrieveToken,
    SupportLevel,
    SupportToken,
    UsefulnessToken,
)
from src.reasoning.selfrag.verifier import SelfRAGVerifier


@pytest.fixture
def mock_llm_pass() -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.generate = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps({
                "retrieve": {"decision": "No", "reason": "충분"},
                "is_relevant": [{"source_idx": 1, "decision": "Relevant", "reason": "관련됨"}],
                "is_supported": {"level": "Fully", "supported_parts": ["전체"], "unsupported_parts": []},
                "is_useful": {"score": 5, "reason": "유용"},
                "final_verdict": "pass",
                "adjusted_answer": None,
            }),
            provider="mock",
            model="mock",
            input_tokens=100,
            output_tokens=200,
            elapsed_ms=500.0,
        )
    )
    return client


class TestSelfRAGVerifier:
    async def test_verification_pass(self, mock_llm_pass: MagicMock) -> None:
        verifier = SelfRAGVerifier(llm_client=mock_llm_pass)
        result = await verifier.verify(
            query="테스트 질문",
            answer="테스트 답변",
            evidence=[{"content": "증거", "source": "test"}],
            current_confidence=0.8,
        )

        assert result.passed is True
        assert result.confidence > 0
        assert result.support_level == "Fully"
        assert result.usefulness_score == 5
        assert not result.retrieve_needed

    async def test_verification_with_warnings(self, mock_llm_pass: MagicMock) -> None:
        mock_llm_pass.generate = AsyncMock(
            return_value=LLMResponse(
                content=json.dumps({
                    "retrieve": {"decision": "Yes", "reason": "추가 데이터 필요"},
                    "is_relevant": [{"source_idx": 1, "decision": "Irrelevant", "reason": "무관"}],
                    "is_supported": {"level": "Partially", "supported_parts": ["일부"], "unsupported_parts": ["나머지"]},
                    "is_useful": {"score": 2, "reason": "부족"},
                    "final_verdict": "needs_retry",
                    "adjusted_answer": "수정된 답변",
                }),
                provider="mock",
                model="mock",
                input_tokens=100,
                output_tokens=200,
                elapsed_ms=500.0,
            )
        )

        verifier = SelfRAGVerifier(llm_client=mock_llm_pass)
        result = await verifier.verify(
            query="테스트",
            answer="답변",
            evidence=[{"content": "증거", "source": "test"}],
            current_confidence=0.5,
        )

        assert result.retrieve_needed is True
        assert len(result.warnings) > 0


class TestReflectionTokens:
    def test_retrieve_token(self) -> None:
        token = RetrieveToken(decision=RetrieveDecision.YES, reason="추가 검색 필요")
        assert token.needs_retrieval is True

        token_no = RetrieveToken(decision=RetrieveDecision.NO, reason="충분")
        assert token_no.needs_retrieval is False

    def test_relevance_token(self) -> None:
        token = RelevanceToken(source_idx=1, decision=RelevanceDecision.RELEVANT, reason="관련")
        assert token.is_relevant is True

    def test_support_token(self) -> None:
        token = SupportToken(level=SupportLevel.FULLY, supported_parts=["전체"], unsupported_parts=[])
        assert token.is_supported is True

        token_no = SupportToken(level=SupportLevel.NO, supported_parts=[], unsupported_parts=["전체"])
        assert token_no.is_supported is False

    def test_usefulness_token(self) -> None:
        token = UsefulnessToken(score=4, reason="유용")
        assert token.is_useful is True

        token_low = UsefulnessToken(score=2, reason="부족")
        assert token_low.is_useful is False
