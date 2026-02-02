"""Tests for GraphRepository (Neo4j)."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data.models.entities import GraphNode, GraphRelationship
from src.data.repositories.graph import GraphRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _AsyncCtxSession:
    """Wrapper that makes a mock session usable as an async context manager.

    The source code calls ``async with await self._get_session() as session:``.
    ``_get_session`` is an async method, so ``await _get_session()`` must yield
    an object that implements ``__aenter__`` / ``__aexit__``.  This thin wrapper
    delegates attribute access to the inner ``AsyncMock`` session while providing
    the required async-context-manager protocol.
    """

    def __init__(self, inner: AsyncMock) -> None:
        self._inner = inner

    async def __aenter__(self) -> AsyncMock:
        return self._inner

    async def __aexit__(self, *args: object) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.neo4j_uri = "bolt://localhost:7687"
    settings.neo4j_user = "neo4j"
    settings.neo4j_password = "test-password"
    return settings


@pytest.fixture
def mock_session():
    """Build an AsyncMock that behaves like a Neo4j AsyncSession."""
    return AsyncMock()


@pytest.fixture
def repo(mock_settings, mock_session):
    """Create a GraphRepository with the Neo4j driver fully mocked.

    The key challenge is the pattern ``async with await self._get_session()``.
    We patch ``_get_session`` to be an async function that returns a wrapper
    supporting the async-context-manager protocol.
    """
    with (
        patch("src.data.repositories.graph.get_settings", return_value=mock_settings),
        patch("src.data.repositories.graph.AsyncGraphDatabase") as MockDriverClass,
    ):
        mock_driver_instance = AsyncMock()
        MockDriverClass.driver.return_value = mock_driver_instance

        repository = GraphRepository()

        # Patch _get_session so ``await repo._get_session()`` returns the wrapper
        ctx_session = _AsyncCtxSession(mock_session)

        async def _fake_get_session():
            return ctx_session

        repository._get_session = _fake_get_session  # type: ignore[assignment]
        yield repository


@pytest.fixture
def sample_node() -> GraphNode:
    return GraphNode(id="node-1", type="metric", name="cpu_usage", properties={"unit": "%"})


@pytest.fixture
def sample_relationship() -> GraphRelationship:
    return GraphRelationship(
        source_id="node-1",
        target_id="node-2",
        type="causes",
        confidence=0.9,
        properties={"lag_minutes": 5},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCreateNode:
    async def test_create_node_runs_cypher(
        self, repo: GraphRepository, mock_session: AsyncMock, sample_node: GraphNode
    ) -> None:
        """create_node executes a CREATE cypher query and returns the same node."""
        result = await repo.create_node(sample_node)

        assert result.id == sample_node.id
        assert result.name == "cpu_usage"
        mock_session.run.assert_awaited_once()

        # Inspect the Cypher query text
        call_args = mock_session.run.call_args
        cypher = call_args.args[0]
        assert "CREATE" in cypher
        assert ":metric" in cypher

    async def test_create_node_passes_properties(
        self, repo: GraphRepository, mock_session: AsyncMock, sample_node: GraphNode
    ) -> None:
        """Node properties dict is forwarded to the Cypher parameters."""
        await repo.create_node(sample_node)

        call_kwargs = mock_session.run.call_args.kwargs
        assert call_kwargs["properties"] == {"unit": "%"}
        assert call_kwargs["id"] == "node-1"
        assert call_kwargs["name"] == "cpu_usage"


class TestGetNode:
    async def test_get_existing_node(
        self, repo: GraphRepository, mock_session: AsyncMock
    ) -> None:
        """get_node returns a dict with type extracted from labels."""
        mock_record = {
            "n": {"id": "node-1", "name": "cpu_usage"},
            "labels": ["metric"],
        }
        mock_result = AsyncMock()
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result

        node_data = await repo.get_node("node-1")

        assert node_data is not None
        assert node_data["id"] == "node-1"
        assert node_data["type"] == "metric"

    async def test_get_nonexistent_node(
        self, repo: GraphRepository, mock_session: AsyncMock
    ) -> None:
        """get_node returns None when no match is found."""
        mock_result = AsyncMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        assert await repo.get_node("missing-id") is None


class TestDeleteNode:
    async def test_delete_existing_node(
        self, repo: GraphRepository, mock_session: AsyncMock
    ) -> None:
        """delete_node returns True when a node was actually deleted."""
        mock_record = {"deleted": 1}
        mock_result = AsyncMock()
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result

        deleted = await repo.delete_node("node-1")

        assert deleted is True
        cypher = mock_session.run.call_args.args[0]
        assert "DETACH DELETE" in cypher

    async def test_delete_nonexistent_node(
        self, repo: GraphRepository, mock_session: AsyncMock
    ) -> None:
        """delete_node returns False when no node matches the given id."""
        mock_record = {"deleted": 0}
        mock_result = AsyncMock()
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result

        assert await repo.delete_node("ghost") is False


class TestCreateRelationship:
    async def test_create_relationship_runs_cypher(
        self,
        repo: GraphRepository,
        mock_session: AsyncMock,
        sample_relationship: GraphRelationship,
    ) -> None:
        """create_relationship builds correct MATCH/CREATE cypher."""
        result = await repo.create_relationship(sample_relationship)

        assert result.source_id == "node-1"
        assert result.target_id == "node-2"
        mock_session.run.assert_awaited_once()

        call_kwargs = mock_session.run.call_args.kwargs
        assert call_kwargs["source_id"] == "node-1"
        assert call_kwargs["target_id"] == "node-2"
        assert call_kwargs["confidence"] == 0.9


class TestTraverse:
    async def test_traverse_default_params(
        self, repo: GraphRepository, mock_session: AsyncMock
    ) -> None:
        """traverse uses default MAX_HOPS and no relationship type filter."""
        # Build an async iterator that yields nothing
        async def _empty_aiter():
            return
            yield  # noqa: make this an async generator

        mock_session.run.return_value = _empty_aiter()

        paths = await repo.traverse(start_node_id="node-1")

        assert paths == []
        cypher = mock_session.run.call_args.args[0]
        # Default MAX_HOPS is 5 (from constants)
        assert "*1..5" in cypher

    async def test_traverse_with_relationship_types(
        self, repo: GraphRepository, mock_session: AsyncMock
    ) -> None:
        """Relationship type filter is injected into the Cypher pattern."""
        async def _empty_aiter():
            return
            yield

        mock_session.run.return_value = _empty_aiter()

        await repo.traverse(
            start_node_id="node-1",
            relationship_types=["causes", "correlates"],
        )

        cypher = mock_session.run.call_args.args[0]
        assert ":CAUSES|CORRELATES" in cypher

    async def test_traverse_returns_paths(
        self, repo: GraphRepository, mock_session: AsyncMock
    ) -> None:
        """traverse correctly collects path records from the async result."""
        record1 = {
            "nodes": [{"id": "a", "name": "A", "labels": ["metric"]}],
            "rels": [{"type": "CAUSES", "confidence": 0.8}],
            "hops": 1,
        }
        record2 = {
            "nodes": [
                {"id": "a", "name": "A", "labels": ["metric"]},
                {"id": "b", "name": "B", "labels": ["event"]},
            ],
            "rels": [
                {"type": "CAUSES", "confidence": 0.8},
                {"type": "CORRELATES", "confidence": 0.6},
            ],
            "hops": 2,
        }

        async def _records_aiter():
            for rec in [record1, record2]:
                yield rec

        mock_session.run.return_value = _records_aiter()

        paths = await repo.traverse(start_node_id="a", max_hops=3)

        assert len(paths) == 2
        assert paths[0]["hops"] == 1
        assert paths[1]["hops"] == 2
        assert paths[1]["nodes"][1]["id"] == "b"


class TestFindCausalChain:
    async def test_delegates_to_traverse(self, repo: GraphRepository) -> None:
        """find_causal_chain calls traverse with 'causes' type and min_confidence=0.5."""
        repo.traverse = AsyncMock(return_value=[])

        await repo.find_causal_chain(start_node_id="node-1", max_hops=3)

        repo.traverse.assert_awaited_once_with(
            start_node_id="node-1",
            max_hops=3,
            relationship_types=["causes"],
            min_confidence=0.5,
        )


class TestGetCommunitySummary:
    async def test_existing_community(
        self, repo: GraphRepository, mock_session: AsyncMock
    ) -> None:
        """Returns community dict when members exist."""
        mock_record = {
            "members": [{"id": "n1", "name": "Node 1", "labels": ["metric"]}],
            "size": 1,
        }
        mock_result = AsyncMock()
        mock_result.single.return_value = mock_record
        mock_session.run.return_value = mock_result

        summary = await repo.get_community_summary("comm-1")

        assert summary is not None
        assert summary["community_id"] == "comm-1"
        assert summary["size"] == 1
        assert len(summary["members"]) == 1

    async def test_nonexistent_community(
        self, repo: GraphRepository, mock_session: AsyncMock
    ) -> None:
        """Returns None when the community query yields no record."""
        mock_result = AsyncMock()
        mock_result.single.return_value = None
        mock_session.run.return_value = mock_result

        assert await repo.get_community_summary("ghost-comm") is None
