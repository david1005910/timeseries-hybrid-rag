"""Tests for ExtractorAgent: entity/relationship extraction from text."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.base import AgentContext, AgentResult, AgentStatus
from src.agents.extractor.agent import ExtractorAgent
from src.data.repositories.graph import GraphRepository
from src.llm.client import LLMClient, LLMResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        provider="mock",
        model="mock-model",
        input_tokens=100,
        output_tokens=200,
        elapsed_ms=400.0,
    )


@pytest.fixture
def extraction_json() -> dict[str, Any]:
    """Canonical extraction output from the LLM."""
    return {
        "entities": [
            {"name": "batch_job", "type": "event", "properties": {"schedule": "daily"}},
            {"name": "cpu_spike", "type": "metric", "properties": {"threshold": 90}},
        ],
        "relationships": [
            {"source": "batch_job", "target": "cpu_spike", "type": "causes", "confidence": 0.9},
        ],
    }


@pytest.fixture
def mock_llm(extraction_json: dict[str, Any]) -> MagicMock:
    client = MagicMock(spec=LLMClient)
    client.generate = AsyncMock(
        return_value=_make_llm_response(json.dumps(extraction_json))
    )
    return client


@pytest.fixture
def mock_graph_repo() -> MagicMock:
    repo = MagicMock(spec=GraphRepository)
    repo.create_node = AsyncMock(side_effect=lambda node: node)
    repo.create_relationship = AsyncMock(side_effect=lambda rel: rel)
    return repo


@pytest.fixture
def extractor(mock_llm: MagicMock, mock_graph_repo: MagicMock) -> ExtractorAgent:
    return ExtractorAgent(llm_client=mock_llm, graph_repo=mock_graph_repo)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractorAgent:

    async def test_execute_with_valid_documents(
        self,
        extractor: ExtractorAgent,
        mock_llm: MagicMock,
        mock_graph_repo: MagicMock,
    ) -> None:
        """execute() should extract entities/relationships and persist them to the graph."""
        context = AgentContext(
            query="엔티티 추출",
            previous_results={
                "documents": [
                    {
                        "id": "doc-1",
                        "content": "대규모 배치 작업이 실행되어 CPU 사용률이 95%까지 급증하는 현상이 발생했습니다.",
                        "source_type": "vector",
                    }
                ]
            },
        )

        result = await extractor.execute(context)

        assert result.status == AgentStatus.SUCCESS
        assert result.data["entity_count"] == 2
        assert result.data["relationship_count"] == 1

        # Graph repo should have been called to create 2 nodes and 1 relationship
        assert mock_graph_repo.create_node.await_count == 2
        assert mock_graph_repo.create_relationship.await_count == 1

        entities_names = [e["name"] for e in result.data["entities"]]
        assert "batch_job" in entities_names
        assert "cpu_spike" in entities_names

    async def test_execute_with_empty_documents(
        self,
        extractor: ExtractorAgent,
        mock_llm: MagicMock,
    ) -> None:
        """When no documents are provided, result should be empty with SUCCESS status."""
        context = AgentContext(
            query="빈 문서 테스트",
            previous_results={"documents": []},
        )

        result = await extractor.execute(context)

        assert result.status == AgentStatus.SUCCESS
        assert result.data["entities"] == []
        assert result.data["relationships"] == []
        mock_llm.generate.assert_not_awaited()

    async def test_execute_skips_short_content(
        self,
        extractor: ExtractorAgent,
        mock_llm: MagicMock,
    ) -> None:
        """Documents with content shorter than 20 characters should be skipped."""
        context = AgentContext(
            query="짧은 문서",
            previous_results={
                "documents": [
                    {"id": "doc-short", "content": "짧은 텍스트", "source_type": "vector"},
                ]
            },
        )

        result = await extractor.execute(context)

        assert result.status == AgentStatus.SUCCESS
        assert result.data["entity_count"] == 0
        mock_llm.generate.assert_not_awaited()

    async def test_extract_from_text_json_parsing(
        self,
        mock_graph_repo: MagicMock,
    ) -> None:
        """_extract_from_text should correctly parse a valid JSON response."""
        expected = {
            "entities": [{"name": "server", "type": "entity", "properties": {}}],
            "relationships": [],
        }
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.generate = AsyncMock(
            return_value=_make_llm_response(json.dumps(expected))
        )

        extractor = ExtractorAgent(llm_client=mock_llm, graph_repo=mock_graph_repo)
        extraction = await extractor._extract_from_text("서버 모니터링 데이터를 분석합니다.")

        assert extraction == expected

    async def test_extract_from_text_json_with_surrounding_text(
        self,
        mock_graph_repo: MagicMock,
    ) -> None:
        """_extract_from_text should extract JSON embedded in surrounding prose."""
        inner_json = {"entities": [{"name": "network", "type": "concept", "properties": {}}], "relationships": []}
        raw_content = f"Here is the extraction result:\n{json.dumps(inner_json)}\nDone."

        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.generate = AsyncMock(return_value=_make_llm_response(raw_content))

        extractor = ExtractorAgent(llm_client=mock_llm, graph_repo=mock_graph_repo)
        extraction = await extractor._extract_from_text("네트워크 장애가 발생했습니다. 원인을 분석해주세요.")

        assert extraction["entities"][0]["name"] == "network"

    async def test_extract_from_text_malformed_response_returns_empty(
        self,
        mock_graph_repo: MagicMock,
    ) -> None:
        """If the LLM returns text with no JSON at all, return empty entities/relationships."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.generate = AsyncMock(
            return_value=_make_llm_response("이것은 JSON이 아닌 일반 텍스트 응답입니다.")
        )

        extractor = ExtractorAgent(llm_client=mock_llm, graph_repo=mock_graph_repo)
        extraction = await extractor._extract_from_text("테스트 텍스트입니다. 엔티티를 추출하세요.")

        assert extraction == {"entities": [], "relationships": []}

    async def test_execute_handles_llm_exception_gracefully(
        self,
        mock_graph_repo: MagicMock,
    ) -> None:
        """If the LLM raises an exception during extraction, that document is skipped."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM timeout"))

        extractor = ExtractorAgent(llm_client=mock_llm, graph_repo=mock_graph_repo)
        context = AgentContext(
            query="예외 테스트",
            previous_results={
                "documents": [
                    {"id": "doc-err", "content": "충분히 긴 텍스트 콘텐츠입니다. 에러 발생 시나리오 테스트용입니다.", "source_type": "vector"},
                ]
            },
        )

        result = await extractor.execute(context)

        assert result.status == AgentStatus.SUCCESS
        assert result.data["entity_count"] == 0
        assert result.data["relationship_count"] == 0

    async def test_execute_limits_documents_to_ten(
        self,
        mock_graph_repo: MagicMock,
        extraction_json: dict[str, Any],
    ) -> None:
        """execute() should process at most 10 documents even if more are provided."""
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.generate = AsyncMock(
            return_value=_make_llm_response(json.dumps(extraction_json))
        )

        extractor = ExtractorAgent(llm_client=mock_llm, graph_repo=mock_graph_repo)
        docs = [
            {"id": f"doc-{i}", "content": f"문서 내용 번호 {i}. 이 문서에는 엔티티가 포함되어 있습니다.", "source_type": "vector"}
            for i in range(15)
        ]
        context = AgentContext(
            query="문서 제한 테스트",
            previous_results={"documents": docs},
        )

        await extractor.execute(context)

        # Only 10 documents should trigger LLM calls
        assert mock_llm.generate.await_count == 10
