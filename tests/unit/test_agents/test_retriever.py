"""Tests for RetrieverAgent: multi-source parallel retrieval."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.base import AgentContext, AgentResult, AgentStatus
from src.agents.retriever.agent import RetrieverAgent
from src.data.repositories.graph import GraphRepository
from src.data.repositories.timeseries import TimeseriesRepository
from src.data.repositories.vector import VectorRepository
from src.llm.embeddings import EmbeddingService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_embedding_service() -> MagicMock:
    svc = MagicMock(spec=EmbeddingService)
    svc.embed = AsyncMock(return_value=[0.1] * 1536)
    return svc


@pytest.fixture
def mock_vector_repo() -> MagicMock:
    repo = MagicMock(spec=VectorRepository)
    repo.search = AsyncMock(
        return_value=[
            {
                "id": "v-1",
                "content": "CPU 사용률이 95%로 급증",
                "source": "monitoring",
                "source_type": "vector",
                "score": 0.92,
                "metadata_json": json.dumps({"host": "server-01"}),
            },
            {
                "id": "v-2",
                "content": "메모리 사용률 88%",
                "source": "monitoring",
                "source_type": "vector",
                "score": 0.85,
                "metadata_json": json.dumps({"host": "server-01"}),
            },
        ]
    )
    return repo


@pytest.fixture
def mock_graph_repo() -> MagicMock:
    repo = MagicMock(spec=GraphRepository)
    repo.traverse = AsyncMock(
        return_value=[
            {
                "nodes": [
                    {"name": "batch_job", "labels": ["event"]},
                    {"name": "cpu_spike", "labels": ["metric"]},
                ],
                "relationships": [{"type": "CAUSES", "confidence": 0.9}],
                "hops": 1,
            }
        ]
    )
    return repo


@pytest.fixture
def mock_timeseries_repo() -> MagicMock:
    repo = MagicMock(spec=TimeseriesRepository)
    repo.query_metrics = AsyncMock(
        return_value=[
            {"time": "2025-01-31T14:30:00", "measurement": "cpu", "field": "usage", "value": 45.0, "tags": {}},
            {"time": "2025-01-31T14:32:00", "measurement": "cpu", "field": "usage", "value": 95.0, "tags": {}},
        ]
    )
    return repo


@pytest.fixture
def retriever(
    mock_timeseries_repo: MagicMock,
    mock_vector_repo: MagicMock,
    mock_graph_repo: MagicMock,
    mock_embedding_service: MagicMock,
) -> RetrieverAgent:
    return RetrieverAgent(
        timeseries_repo=mock_timeseries_repo,
        vector_repo=mock_vector_repo,
        graph_repo=mock_graph_repo,
        embedding_service=mock_embedding_service,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRetrieverAgent:

    async def test_execute_vector_only(
        self,
        retriever: RetrieverAgent,
        mock_vector_repo: MagicMock,
        mock_graph_repo: MagicMock,
        mock_timeseries_repo: MagicMock,
    ) -> None:
        """When only 'vector' is in sources, only vector search should be called."""
        context = AgentContext(
            query="CPU 급증 원인",
            options={
                "retrieval_plan": {"sources": ["vector"]},
                "top_k": 5,
            },
        )

        result = await retriever.execute(context)

        assert result.status == AgentStatus.SUCCESS
        assert result.data["total_count"] == 2
        mock_vector_repo.search.assert_awaited_once()
        mock_graph_repo.traverse.assert_not_awaited()
        mock_timeseries_repo.query_metrics.assert_not_awaited()

    async def test_execute_all_sources(
        self,
        retriever: RetrieverAgent,
        mock_vector_repo: MagicMock,
        mock_graph_repo: MagicMock,
        mock_timeseries_repo: MagicMock,
    ) -> None:
        """With all three sources enabled, all repos should be queried."""
        context = AgentContext(
            query="CPU 급증 원인",
            options={
                "retrieval_plan": {
                    "sources": ["vector", "graph", "timeseries"],
                    "start_node_id": "node-cpu",
                    "timeseries_params": {"measurement": "cpu"},
                },
                "top_k": 10,
                "max_hops": 3,
            },
        )

        result = await retriever.execute(context)

        assert result.status == AgentStatus.SUCCESS
        # 2 vector + 1 graph + 2 timeseries = 5
        assert result.data["total_count"] == 5
        assert result.data["source_counts"]["vector"] == 2
        assert result.data["source_counts"]["graph"] == 1
        assert result.data["source_counts"]["timeseries"] == 2
        mock_vector_repo.search.assert_awaited_once()
        mock_graph_repo.traverse.assert_awaited_once()
        mock_timeseries_repo.query_metrics.assert_awaited_once()

    async def test_execute_handles_failed_source(
        self,
        mock_vector_repo: MagicMock,
        mock_graph_repo: MagicMock,
        mock_timeseries_repo: MagicMock,
        mock_embedding_service: MagicMock,
    ) -> None:
        """If one source fails, others should still return results."""
        # Make graph repo raise an exception
        mock_graph_repo.traverse = AsyncMock(side_effect=Exception("Neo4j connection refused"))

        retriever = RetrieverAgent(
            timeseries_repo=mock_timeseries_repo,
            vector_repo=mock_vector_repo,
            graph_repo=mock_graph_repo,
            embedding_service=mock_embedding_service,
        )

        context = AgentContext(
            query="CPU 급증 원인",
            options={
                "retrieval_plan": {
                    "sources": ["vector", "graph", "timeseries"],
                    "start_node_id": "node-cpu",
                    "timeseries_params": {"measurement": "cpu"},
                },
            },
        )

        result = await retriever.execute(context)

        assert result.status == AgentStatus.SUCCESS
        # vector (2) + timeseries (2) = 4 (graph failed)
        assert result.data["total_count"] == 4
        # graph entry should have an error marker
        assert "error" in result.data["source_counts"] or result.data["source_counts"].get("graph") == 0

    async def test_merge_results_sorted_by_relevance(
        self, retriever: RetrieverAgent
    ) -> None:
        """_merge_results should sort items by relevance_score descending."""
        results: dict[str, Any] = {
            "vector": {
                "items": [
                    {"id": "v1", "relevance_score": 0.6},
                    {"id": "v2", "relevance_score": 0.9},
                ]
            },
            "graph": {
                "items": [
                    {"id": "g1", "relevance_score": 0.8},
                ]
            },
            "timeseries": {
                "items": [
                    {"id": "t1", "relevance_score": 0.7},
                ]
            },
        }

        merged = retriever._merge_results(results)

        scores = [item["relevance_score"] for item in merged]
        assert scores == sorted(scores, reverse=True)
        assert merged[0]["id"] == "v2"
        assert merged[-1]["id"] == "v1"

    async def test_merge_results_handles_error_entries(
        self, retriever: RetrieverAgent
    ) -> None:
        """_merge_results should gracefully handle entries with error and items."""
        results: dict[str, Any] = {
            "vector": {"items": [{"id": "v1", "relevance_score": 0.9}]},
            "graph": {"error": "Neo4j down", "items": []},
        }

        merged = retriever._merge_results(results)

        assert len(merged) == 1
        assert merged[0]["id"] == "v1"

    def test_format_path(self) -> None:
        """_format_path should produce a human-readable path description."""
        path = {
            "nodes": [
                {"name": "batch_job"},
                {"name": "cpu_spike"},
                {"name": "alert"},
            ],
            "relationships": [
                {"type": "CAUSES"},
                {"type": "TRIGGERS"},
            ],
            "hops": 2,
        }

        result = RetrieverAgent._format_path(path)

        assert "[batch_job]" in result
        assert "--CAUSES-->" in result
        assert "[cpu_spike]" in result
        assert "--TRIGGERS-->" in result
        assert "[alert]" in result

    def test_format_path_empty(self) -> None:
        """_format_path should handle a path with no nodes gracefully."""
        path: dict[str, Any] = {"nodes": [], "relationships": []}
        result = RetrieverAgent._format_path(path)
        assert result == ""

    async def test_execute_graph_without_start_node(
        self,
        retriever: RetrieverAgent,
        mock_graph_repo: MagicMock,
    ) -> None:
        """Graph search without start_node_id should return empty items."""
        context = AgentContext(
            query="그래프 탐색",
            options={
                "retrieval_plan": {
                    "sources": ["graph"],
                    # no start_node_id
                },
            },
        )

        result = await retriever.execute(context)

        assert result.status == AgentStatus.SUCCESS
        assert result.data["source_counts"]["graph"] == 0
        mock_graph_repo.traverse.assert_not_awaited()
