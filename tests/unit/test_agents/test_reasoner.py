"""Tests for ReasonerAgent: Chain-of-Thought reasoning with graph integration."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.base import AgentContext, AgentResult, AgentStatus
from src.agents.reasoner.agent import ReasonerAgent
from src.llm.client import LLMClient, LLMResponse


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------

def _make_reasoning_json(**overrides: Any) -> dict[str, Any]:
    """Build a canonical reasoning JSON payload."""
    base = {
        "reasoning_steps": [
            {"step": 1, "action": "분석", "description": "CPU 데이터 확인", "evidence_used": ["doc-1"]},
            {"step": 2, "action": "추론", "description": "배치 작업과 상관관계 확인", "evidence_used": ["doc-1", "path-0"]},
        ],
        "answer": "배치 작업이 CPU 급증의 원인입니다.",
        "confidence": 0.88,
        "causal_chain": ["batch_job -> cpu_spike", "cpu_spike -> alert"],
        "uncertainties": ["메모리 영향 불확실"],
    }
    base.update(overrides)
    return base


def _make_llm_response(content: str, input_tokens: int = 500, output_tokens: int = 300) -> LLMResponse:
    return LLMResponse(
        content=content,
        provider="mock",
        model="mock-model",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_ms=600.0,
    )


@pytest.fixture
def reasoning_json() -> dict[str, Any]:
    return _make_reasoning_json()


@pytest.fixture
def mock_llm(reasoning_json: dict[str, Any]) -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.generate = AsyncMock(
        return_value=_make_llm_response(json.dumps(reasoning_json))
    )
    return client


@pytest.fixture
def reasoner(mock_llm: MagicMock) -> ReasonerAgent:
    return ReasonerAgent(llm_client=mock_llm)


@pytest.fixture
def mixed_documents() -> list[dict[str, Any]]:
    """Documents spanning all three source types."""
    return [
        {
            "id": "doc-1",
            "content": "CPU 사용률이 95%까지 급증했습니다.",
            "source_type": "vector",
            "relevance_score": 0.92,
        },
        {
            "id": "doc-2",
            "content": "메모리 사용량이 88%입니다.",
            "source_type": "vector",
            "relevance_score": 0.85,
        },
        {
            "id": "path-0",
            "content": "[batch_job] --CAUSES--> [cpu_spike]",
            "source_type": "graph",
            "relevance_score": 0.8,
        },
        {
            "id": "ts-0",
            "content": json.dumps({"time": "2025-01-31T14:30:00", "field": "cpu_usage", "value": 45.0}),
            "source_type": "timeseries",
            "relevance_score": 0.7,
        },
        {
            "id": "ts-1",
            "content": json.dumps({"time": "2025-01-31T14:32:00", "field": "cpu_usage", "value": 95.0}),
            "source_type": "timeseries",
            "relevance_score": 0.7,
        },
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestReasonerAgent:

    async def test_execute_with_mixed_evidence(
        self,
        reasoner: ReasonerAgent,
        mock_llm: MagicMock,
        mixed_documents: list[dict[str, Any]],
    ) -> None:
        """execute() with all three source types should produce a complete result."""
        context = AgentContext(
            query="CPU 급증 원인 분석",
            previous_results={"documents": mixed_documents},
        )

        result = await reasoner.execute(context)

        assert result.status == AgentStatus.SUCCESS
        assert result.data["answer"] == "배치 작업이 CPU 급증의 원인입니다."
        assert result.data["confidence"] == pytest.approx(0.88)
        assert len(result.data["reasoning_steps"]) == 2
        assert len(result.data["causal_chain"]) == 2
        assert result.data["sources_used"] == 5

        # Metadata should contain token/cost information
        assert result.metadata["tokens_used"] == 800  # 500 + 300
        assert "cost_usd" in result.metadata  # provider="mock" yields 0.0
        assert result.metadata["llm_elapsed_ms"] == 600.0

        # LLM should have been called exactly once
        mock_llm.generate.assert_awaited_once()

    async def test_execute_with_no_evidence(
        self,
        mock_llm: MagicMock,
    ) -> None:
        """When no documents are provided, the prompt should include default empty markers."""
        reasoning_no_evidence = _make_reasoning_json(
            answer="충분한 증거가 없습니다.", confidence=0.3,
            causal_chain=[], uncertainties=["모든 증거 부족"],
        )
        mock_llm.generate = AsyncMock(
            return_value=_make_llm_response(json.dumps(reasoning_no_evidence))
        )

        reasoner = ReasonerAgent(llm_client=mock_llm)
        context = AgentContext(
            query="알 수 없는 질문",
            previous_results={"documents": []},
        )

        result = await reasoner.execute(context)

        assert result.status == AgentStatus.SUCCESS
        assert result.data["confidence"] == pytest.approx(0.3)
        assert result.data["sources_used"] == 0

        # Verify the prompt was constructed with empty markers
        call_kwargs = mock_llm.generate.call_args
        prompt_text = call_kwargs.kwargs.get("prompt") or call_kwargs.args[0]
        assert "검색된 문서 없음" in prompt_text
        assert "그래프 경로 없음" in prompt_text
        assert "시계열 데이터 없음" in prompt_text

    async def test_json_parsing_fallback_extracts_embedded_json(
        self,
        mock_llm: MagicMock,
    ) -> None:
        """When LLM wraps JSON in prose, the fallback parser should extract it."""
        inner = _make_reasoning_json(answer="추출된 JSON 답변", confidence=0.75)
        raw_content = f"Here is my analysis:\n{json.dumps(inner)}\nEnd of analysis."

        mock_llm.generate = AsyncMock(return_value=_make_llm_response(raw_content))

        reasoner = ReasonerAgent(llm_client=mock_llm)
        context = AgentContext(
            query="JSON 파싱 폴백 테스트",
            previous_results={"documents": []},
        )

        result = await reasoner.execute(context)

        assert result.data["answer"] == "추출된 JSON 답변"
        assert result.data["confidence"] == pytest.approx(0.75)

    async def test_json_parsing_total_failure_returns_raw_content(
        self,
        mock_llm: MagicMock,
    ) -> None:
        """When LLM returns no JSON at all, the raw content becomes the answer."""
        raw_text = "이것은 JSON이 전혀 포함되지 않은 응답입니다."
        mock_llm.generate = AsyncMock(return_value=_make_llm_response(raw_text))

        reasoner = ReasonerAgent(llm_client=mock_llm)
        context = AgentContext(
            query="파싱 실패 테스트",
            previous_results={"documents": []},
        )

        result = await reasoner.execute(context)

        assert result.data["answer"] == raw_text
        assert result.data["confidence"] == 0.5
        assert "LLM 응답 파싱 실패" in result.data["uncertainties"]

    def test_format_evidence(self) -> None:
        """_format_evidence should number documents and include score and content."""
        docs = [
            {"content": "첫 번째 문서 내용", "relevance_score": 0.9},
            {"content": "두 번째 문서 내용", "relevance_score": 0.8},
        ]

        result = ReasonerAgent._format_evidence(docs)

        assert "[1]" in result
        assert "(score: 0.90)" in result
        assert "첫 번째 문서 내용" in result
        assert "[2]" in result
        assert "(score: 0.80)" in result
        assert "두 번째 문서 내용" in result

    def test_format_evidence_empty(self) -> None:
        """_format_evidence with an empty list should return empty string."""
        result = ReasonerAgent._format_evidence([])
        assert result == ""

    def test_format_graph_paths(self) -> None:
        """_format_graph_paths should number paths."""
        docs = [
            {"content": "[A] --CAUSES--> [B]"},
            {"content": "[B] --TRIGGERS--> [C]"},
        ]

        result = ReasonerAgent._format_graph_paths(docs)

        assert "경로 1:" in result
        assert "[A] --CAUSES--> [B]" in result
        assert "경로 2:" in result

    def test_format_timeseries(self) -> None:
        """_format_timeseries should join timeseries content with newlines."""
        docs = [
            {"content": '{"time":"T1","value":45.0}'},
            {"content": '{"time":"T2","value":95.0}'},
        ]

        result = ReasonerAgent._format_timeseries(docs)

        assert '{"time":"T1","value":45.0}' in result
        assert '{"time":"T2","value":95.0}' in result
        # Should be separated by newline
        lines = result.strip().split("\n")
        assert len(lines) == 2

    def test_format_timeseries_limits_to_ten(self) -> None:
        """_format_timeseries should include at most 10 documents."""
        docs = [{"content": f"ts-{i}"} for i in range(15)]

        result = ReasonerAgent._format_timeseries(docs)
        lines = [line for line in result.strip().split("\n") if line]
        assert len(lines) == 10
