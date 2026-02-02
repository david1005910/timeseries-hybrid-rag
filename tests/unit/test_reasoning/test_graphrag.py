"""Tests for GraphRAG Engine."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.repositories.graph import GraphRepository
from src.llm.client import LLMClient, LLMResponse
from src.reasoning.graphrag.engine import GraphRAGEngine, GraphRAGResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm() -> MagicMock:
    """Mock LLM client that returns a valid traversal strategy JSON."""
    client = MagicMock(spec=LLMClient)
    client.generate = AsyncMock(
        return_value=LLMResponse(
            content=json.dumps({
                "start_entities": ["CPU"],
                "relationship_types": ["causes", "correlates"],
                "max_hops": 3,
                "reasoning": "CPU 관련 인과관계 탐색",
            }),
            provider="mock",
            model="mock-model",
            input_tokens=80,
            output_tokens=120,
            elapsed_ms=200.0,
        )
    )
    return client


@pytest.fixture
def mock_graph_repo() -> MagicMock:
    """Mock GraphRepository with traverse and find_causal_chain stubs."""
    repo = MagicMock(spec=GraphRepository)
    repo.traverse = AsyncMock(return_value=[
        {
            "nodes": [
                {"name": "CPU", "id": "n1"},
                {"name": "BatchJob", "id": "n2"},
            ],
            "relationships": [
                {"type": "CAUSES", "confidence": 0.9},
            ],
            "hops": 1,
        },
    ])
    repo.find_causal_chain = AsyncMock(return_value=[
        {
            "nodes": [
                {"name": "CPU", "id": "n1"},
                {"name": "BatchJob", "id": "n2"},
            ],
            "relationships": [
                {"type": "CAUSES", "confidence": 0.85},
            ],
            "hops": 1,
        },
    ])
    return repo


@pytest.fixture
def engine(mock_graph_repo: MagicMock, mock_llm: MagicMock) -> GraphRAGEngine:
    return GraphRAGEngine(graph_repo=mock_graph_repo, llm_client=mock_llm)


# ---------------------------------------------------------------------------
# Helper to build a mock Neo4j record
# ---------------------------------------------------------------------------

def _make_neo4j_record(data: dict[str, Any]) -> MagicMock:
    record = MagicMock()
    record.__getitem__ = lambda self, key: data[key]
    return record


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGraphRAGResult:
    """GraphRAGResult data class tests."""

    def test_attributes(self) -> None:
        result = GraphRAGResult(
            paths=[{"nodes": [], "relationships": [], "hops": 0}],
            explanation="설명 텍스트",
            strategy={"start_entities": ["A"]},
            total_paths=1,
        )
        assert result.total_paths == 1
        assert result.explanation == "설명 텍스트"
        assert len(result.paths) == 1
        assert result.strategy["start_entities"] == ["A"]

    def test_empty_result(self) -> None:
        result = GraphRAGResult(paths=[], explanation="", strategy={}, total_paths=0)
        assert result.total_paths == 0
        assert result.paths == []
        assert result.explanation == ""


class TestGraphRAGEngine:
    """Tests for the GraphRAGEngine async methods."""

    async def test_query_full_pipeline(
        self, engine: GraphRAGEngine, mock_graph_repo: MagicMock, mock_llm: MagicMock,
    ) -> None:
        """query() should plan traversal, traverse the graph, and explain paths."""
        # Mock _find_node_by_name to return a valid node
        with patch.object(
            engine, "_find_node_by_name", new_callable=AsyncMock,
            return_value={"id": "n1", "name": "CPU", "type": "Metric"},
        ):
            # Set up the LLM to return explanation on the second call
            mock_llm.generate = AsyncMock(
                side_effect=[
                    # First call: _plan_traversal
                    LLMResponse(
                        content=json.dumps({
                            "start_entities": ["CPU"],
                            "relationship_types": ["causes"],
                            "max_hops": 3,
                            "reasoning": "CPU 인과관계 탐색",
                        }),
                        provider="mock", model="mock", input_tokens=50,
                        output_tokens=80, elapsed_ms=100.0,
                    ),
                    # Second call: _explain_paths
                    LLMResponse(
                        content="CPU 급증은 BatchJob에 의해 발생했습니다.",
                        provider="mock", model="mock", input_tokens=100,
                        output_tokens=150, elapsed_ms=200.0,
                    ),
                ]
            )

            result = await engine.query("CPU 급증 원인은?")

        assert isinstance(result, GraphRAGResult)
        assert result.total_paths > 0
        assert "BatchJob" in result.explanation or result.explanation != ""
        assert result.strategy["start_entities"] == ["CPU"]
        mock_graph_repo.traverse.assert_awaited_once()

    async def test_query_no_nodes_found(
        self, engine: GraphRAGEngine, mock_graph_repo: MagicMock,
    ) -> None:
        """query() should return empty result when no start nodes are found."""
        with patch.object(
            engine, "_find_node_by_name", new_callable=AsyncMock, return_value=None,
        ):
            result = await engine.query("존재하지 않는 엔티티")

        assert result.total_paths == 0
        assert result.explanation == ""
        mock_graph_repo.traverse.assert_not_awaited()

    async def test_find_causal_chain_found(
        self, engine: GraphRAGEngine, mock_graph_repo: MagicMock,
    ) -> None:
        """find_causal_chain should return chain when start node exists."""
        with patch.object(
            engine, "_find_node_by_name", new_callable=AsyncMock,
            return_value={"id": "n1", "name": "CPU", "type": "Metric"},
        ):
            chain = await engine.find_causal_chain("CPU")

        assert len(chain) == 1
        mock_graph_repo.find_causal_chain.assert_awaited_once_with(
            start_node_id="n1", max_hops=5,
        )

    async def test_find_causal_chain_node_not_found(
        self, engine: GraphRAGEngine, mock_graph_repo: MagicMock,
    ) -> None:
        """find_causal_chain should return [] when start node does not exist."""
        with patch.object(
            engine, "_find_node_by_name", new_callable=AsyncMock, return_value=None,
        ):
            chain = await engine.find_causal_chain("Unknown")

        assert chain == []
        mock_graph_repo.find_causal_chain.assert_not_awaited()

    async def test_plan_traversal_valid_json(
        self, engine: GraphRAGEngine, mock_llm: MagicMock,
    ) -> None:
        """_plan_traversal should parse valid JSON from LLM."""
        strategy = await engine._plan_traversal("CPU 급증 원인은?")

        assert strategy["start_entities"] == ["CPU"]
        assert "causes" in strategy["relationship_types"]
        assert strategy["max_hops"] == 3
        mock_llm.generate.assert_awaited_once()

    async def test_plan_traversal_invalid_json_fallback(
        self, engine: GraphRAGEngine, mock_llm: MagicMock,
    ) -> None:
        """_plan_traversal should return default strategy on malformed JSON."""
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="이것은 유효한 JSON이 아닙니다",
                provider="mock", model="mock", input_tokens=10,
                output_tokens=20, elapsed_ms=50.0,
            )
        )

        strategy = await engine._plan_traversal("테스트 질의")

        assert strategy["start_entities"] == []
        assert strategy["relationship_types"] is None
        assert strategy["max_hops"] == 3
        assert strategy["reasoning"] == "기본 탐색"

    async def test_find_node_by_name_found(
        self, engine: GraphRAGEngine, mock_graph_repo: MagicMock,
    ) -> None:
        """_find_node_by_name should return node dict when Neo4j returns a record."""
        mock_record = _make_neo4j_record({
            "id": "n1", "name": "CPU", "labels": ["Metric"],
        })
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=mock_record)

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_graph_repo._get_session = AsyncMock(return_value=mock_session)

        node = await engine._find_node_by_name("CPU")

        assert node is not None
        assert node["id"] == "n1"
        assert node["name"] == "CPU"
        assert node["type"] == "Metric"

    async def test_find_node_by_name_not_found(
        self, engine: GraphRAGEngine, mock_graph_repo: MagicMock,
    ) -> None:
        """_find_node_by_name should return None when Neo4j returns no record."""
        mock_result = AsyncMock()
        mock_result.single = AsyncMock(return_value=None)

        mock_session = AsyncMock()
        mock_session.run = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_graph_repo._get_session = AsyncMock(return_value=mock_session)

        node = await engine._find_node_by_name("NonExistent")

        assert node is None

    async def test_explain_paths(
        self, engine: GraphRAGEngine, mock_llm: MagicMock,
    ) -> None:
        """_explain_paths should format paths and call LLM for explanation."""
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="CPU 과부하는 배치 작업이 원인입니다.",
                provider="mock", model="mock", input_tokens=80,
                output_tokens=100, elapsed_ms=150.0,
            )
        )

        paths = [
            {
                "nodes": [{"name": "CPU"}, {"name": "BatchJob"}],
                "relationships": [{"type": "CAUSES", "confidence": 0.9}],
                "hops": 1,
            },
        ]

        explanation = await engine._explain_paths("CPU 급증 원인은?", paths)

        assert explanation == "CPU 과부하는 배치 작업이 원인입니다."
        mock_llm.generate.assert_awaited_once()
        call_kwargs = mock_llm.generate.call_args
        prompt_used = call_kwargs.kwargs.get("prompt", call_kwargs.args[0] if call_kwargs.args else "")
        assert "CPU" in prompt_used

    async def test_explain_paths_empty(
        self, engine: GraphRAGEngine, mock_llm: MagicMock,
    ) -> None:
        """_explain_paths with empty list should still call LLM."""
        mock_llm.generate = AsyncMock(
            return_value=LLMResponse(
                content="탐색된 경로가 없습니다.",
                provider="mock", model="mock", input_tokens=20,
                output_tokens=30, elapsed_ms=50.0,
            )
        )

        explanation = await engine._explain_paths("테스트", [])

        assert explanation == "탐색된 경로가 없습니다."
