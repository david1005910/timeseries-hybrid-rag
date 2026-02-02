"""Shared test fixtures and configuration."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.base import AgentContext, AgentResult, AgentStatus
from src.llm.client import LLMClient, LLMResponse


@pytest.fixture
def agent_context() -> AgentContext:
    """기본 에이전트 컨텍스트."""
    return AgentContext(
        query="어제 CPU 급증 원인을 분석해줘",
        session_id="test-session-1",
        user_id="test-user-1",
        options={"max_hops": 5, "include_reasoning": True},
    )


@pytest.fixture
def sample_documents() -> list[dict[str, Any]]:
    """샘플 검색 결과 문서."""
    return [
        {
            "id": "doc-1",
            "content": "14:32에 CPU 사용률이 95%까지 급증했습니다. 대규모 배치 작업이 원인으로 확인되었습니다.",
            "source": "monitoring",
            "source_type": "vector",
            "relevance_score": 0.92,
            "metadata": {"timestamp": "2025-01-31T14:32:00"},
        },
        {
            "id": "doc-2",
            "content": "메모리 사용량이 70%에서 88%로 증가했습니다.",
            "source": "monitoring",
            "source_type": "vector",
            "relevance_score": 0.85,
            "metadata": {"timestamp": "2025-01-31T14:31:00"},
        },
        {
            "id": "ts-1",
            "content": json.dumps({"time": "2025-01-31T14:30:00", "field": "cpu_usage", "value": 45.0}),
            "source": "cpu",
            "source_type": "timeseries",
            "relevance_score": 0.7,
            "metadata": {},
        },
    ]


@pytest.fixture
def mock_llm_client() -> MagicMock:
    """Mock LLM 클라이언트."""
    client = MagicMock(spec=LLMClient)
    client.generate = AsyncMock(
        return_value=LLMResponse(
            content='{"answer": "테스트 답변", "confidence": 0.85, "reasoning_steps": [], "causal_chain": [], "uncertainties": []}',
            provider="mock",
            model="mock-model",
            input_tokens=100,
            output_tokens=200,
            elapsed_ms=500.0,
        )
    )
    return client


@pytest.fixture
def mock_llm_validation_response() -> LLMResponse:
    """Mock 검증 응답."""
    return LLMResponse(
        content=json.dumps({
            "retrieve": {"decision": "No", "reason": "충분한 증거"},
            "is_relevant": [{"source_idx": 1, "decision": "Relevant", "reason": "관련 있음"}],
            "is_supported": {"level": "Fully", "supported_parts": ["답변 전체"], "unsupported_parts": []},
            "is_useful": {"score": 4, "reason": "유용함"},
            "overall_confidence": 0.85,
            "hallucination_flags": [],
            "improvement_suggestions": [],
            "final_verdict": "pass",
            "adjusted_answer": None,
        }),
        provider="mock",
        model="mock-model",
        input_tokens=500,
        output_tokens=300,
        elapsed_ms=800.0,
    )
