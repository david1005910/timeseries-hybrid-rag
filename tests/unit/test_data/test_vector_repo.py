"""Tests for VectorRepository (Milvus)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.data.repositories.vector import COLLECTION_NAME, VectorRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.milvus_host = "localhost"
    settings.milvus_port = 19530
    settings.embedding_dimension = 1536
    return settings


@pytest.fixture
def mock_collection():
    """A mock Milvus Collection that is returned by _ensure_collection."""
    collection = MagicMock()
    collection.load.return_value = None
    collection.flush.return_value = None
    return collection


@pytest.fixture
def repo(mock_settings, mock_collection):
    """Create a VectorRepository with Milvus dependencies mocked."""
    with (
        patch("src.data.repositories.vector.get_settings", return_value=mock_settings),
        patch("src.data.repositories.vector.connections") as mock_connections,
        patch("src.data.repositories.vector.utility") as mock_utility,
        patch("src.data.repositories.vector.Collection", return_value=mock_collection),
    ):
        mock_utility.has_collection.return_value = True

        repository = VectorRepository()
        # Pre-wire so _ensure_collection returns our mock without side effects
        repository._connected = True
        repository._collection = mock_collection
        repository._mock_connections = mock_connections
        repository._mock_utility = mock_utility
        yield repository


@pytest.fixture
def sample_embedding() -> list[float]:
    """A dummy 1536-dim embedding vector (all zeros for testing)."""
    return [0.0] * 1536


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInsert:
    async def test_insert_returns_count(
        self, repo: VectorRepository, mock_collection: MagicMock, sample_embedding: list[float]
    ) -> None:
        """insert delegates to collection.insert and returns the insert count."""
        mock_result = MagicMock()
        mock_result.insert_count = 2
        mock_collection.insert.return_value = mock_result

        count = await repo.insert(
            ids=["id-1", "id-2"],
            embeddings=[sample_embedding, sample_embedding],
            contents=["Hello world", "Test document"],
            sources=["file-a", "file-b"],
            source_types=["document", "document"],
            metadata_jsons=['{}', '{}'],
        )

        assert count == 2
        mock_collection.insert.assert_called_once()
        mock_collection.flush.assert_called_once()

    async def test_insert_passes_correct_data_shape(
        self, repo: VectorRepository, mock_collection: MagicMock, sample_embedding: list[float]
    ) -> None:
        """Inserted data is a list of 6 field lists matching the schema."""
        mock_result = MagicMock()
        mock_result.insert_count = 1
        mock_collection.insert.return_value = mock_result

        await repo.insert(
            ids=["id-1"],
            embeddings=[sample_embedding],
            contents=["content"],
            sources=["src"],
            source_types=["doc"],
            metadata_jsons=['{"key": "value"}'],
        )

        data_arg = mock_collection.insert.call_args.args[0]
        assert len(data_arg) == 6  # id, embedding, content, source, source_type, metadata_json
        assert data_arg[0] == ["id-1"]


class TestSearch:
    async def test_search_without_filter(
        self, repo: VectorRepository, mock_collection: MagicMock, sample_embedding: list[float]
    ) -> None:
        """search without source_type passes expr=None."""
        hit = MagicMock()
        hit.id = "id-1"
        hit.score = 0.95
        hit.entity.get.side_effect = lambda key: {
            "content": "result content",
            "source": "file-a",
            "source_type": "document",
            "metadata_json": "{}",
        }[key]
        mock_collection.search.return_value = [[hit]]

        results = await repo.search(query_embedding=sample_embedding, top_k=5)

        assert len(results) == 1
        assert results[0]["id"] == "id-1"
        assert results[0]["score"] == 0.95
        assert results[0]["content"] == "result content"

        call_kwargs = mock_collection.search.call_args.kwargs
        assert call_kwargs["expr"] is None
        assert call_kwargs["limit"] == 5

    async def test_search_with_source_type_filter(
        self, repo: VectorRepository, mock_collection: MagicMock, sample_embedding: list[float]
    ) -> None:
        """search with source_type generates a filter expression."""
        mock_collection.search.return_value = [[]]

        await repo.search(
            query_embedding=sample_embedding,
            top_k=3,
            source_type="timeseries",
        )

        call_kwargs = mock_collection.search.call_args.kwargs
        assert call_kwargs["expr"] == 'source_type == "timeseries"'

    async def test_search_empty_results(
        self, repo: VectorRepository, mock_collection: MagicMock, sample_embedding: list[float]
    ) -> None:
        """search returns an empty list when Milvus returns no hits."""
        mock_collection.search.return_value = [[]]

        results = await repo.search(query_embedding=sample_embedding)

        assert results == []

    async def test_search_multiple_hits(
        self, repo: VectorRepository, mock_collection: MagicMock, sample_embedding: list[float]
    ) -> None:
        """search maps all hits into dicts correctly."""
        hits = []
        for i in range(3):
            hit = MagicMock()
            hit.id = f"id-{i}"
            hit.score = 0.9 - i * 0.1
            hit.entity.get.side_effect = lambda key, idx=i: {
                "content": f"content-{idx}",
                "source": f"source-{idx}",
                "source_type": "document",
                "metadata_json": "{}",
            }[key]
            hits.append(hit)
        mock_collection.search.return_value = [hits]

        results = await repo.search(query_embedding=sample_embedding, top_k=3)

        assert len(results) == 3
        assert results[0]["id"] == "id-0"
        assert results[2]["content"] == "content-2"


class TestDelete:
    async def test_delete_calls_collection_delete(
        self, repo: VectorRepository, mock_collection: MagicMock
    ) -> None:
        """delete builds an 'id in [...]' expression and calls collection.delete."""
        await repo.delete(ids=["id-1", "id-2"])

        mock_collection.delete.assert_called_once()
        expr_arg = mock_collection.delete.call_args.args[0]
        assert "id-1" in expr_arg
        assert "id-2" in expr_arg


class TestConnection:
    def test_connect_called_once(self, mock_settings: MagicMock) -> None:
        """_connect only establishes a connection if not already connected."""
        with (
            patch("src.data.repositories.vector.get_settings", return_value=mock_settings),
            patch("src.data.repositories.vector.connections") as mock_connections,
            patch("src.data.repositories.vector.utility"),
        ):
            repository = VectorRepository()
            assert repository._connected is False

            repository._connect()
            assert repository._connected is True
            mock_connections.connect.assert_called_once_with(
                "default", host="localhost", port=19530
            )

            # Second call should be a no-op
            repository._connect()
            assert mock_connections.connect.call_count == 1

    def test_close_disconnects(self, mock_settings: MagicMock) -> None:
        """close disconnects and resets the connected flag."""
        with (
            patch("src.data.repositories.vector.get_settings", return_value=mock_settings),
            patch("src.data.repositories.vector.connections") as mock_connections,
            patch("src.data.repositories.vector.utility"),
        ):
            repository = VectorRepository()
            repository._connected = True

            repository.close()

            assert repository._connected is False
            mock_connections.disconnect.assert_called_once_with("default")
