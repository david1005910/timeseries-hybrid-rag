"""Tests for EmbeddingService: single embed, batch embed, and caching."""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.llm.embeddings import EmbeddingService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_settings(**overrides: Any) -> MagicMock:
    defaults = {
        "openai_api_key": "sk-test-key",
        "embedding_model": "text-embedding-3-large",
        "embedding_dimension": 1536,
    }
    defaults.update(overrides)
    settings = MagicMock()
    for key, value in defaults.items():
        setattr(settings, key, value)
    return settings


def _make_embedding_response(embeddings: list[list[float]]) -> MagicMock:
    """Build a mock openai embeddings.create response."""
    data = []
    for emb in embeddings:
        item = MagicMock()
        item.embedding = emb
        data.append(item)
    response = MagicMock()
    response.data = data
    return response


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEmbeddingService:
    """Tests for EmbeddingService with mocked OpenAI client."""

    @patch("src.llm.embeddings.get_settings")
    @patch("src.llm.embeddings.openai.AsyncOpenAI")
    async def test_embed_single_text(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """embed() should return a list[float] from the OpenAI API."""
        mock_get_settings.return_value = _make_settings()

        expected_vector = [0.1, 0.2, 0.3] * 512  # 1536 dims
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(
            return_value=_make_embedding_response([expected_vector])
        )
        mock_openai_cls.return_value = mock_client

        service = EmbeddingService()
        result = await service.embed("테스트 텍스트")

        assert result == expected_vector
        mock_client.embeddings.create.assert_awaited_once_with(
            model="text-embedding-3-large",
            input="테스트 텍스트",
            dimensions=1536,
        )

    @patch("src.llm.embeddings.get_settings")
    @patch("src.llm.embeddings.openai.AsyncOpenAI")
    async def test_embed_batch(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """embed_batch() should return embeddings for all input texts."""
        mock_get_settings.return_value = _make_settings()

        vec_a = [0.1] * 1536
        vec_b = [0.2] * 1536
        vec_c = [0.3] * 1536

        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(
            return_value=_make_embedding_response([vec_a, vec_b, vec_c])
        )
        mock_openai_cls.return_value = mock_client

        service = EmbeddingService()
        results = await service.embed_batch(["text-a", "text-b", "text-c"])

        assert len(results) == 3
        assert results[0] == vec_a
        assert results[1] == vec_b
        assert results[2] == vec_c
        mock_client.embeddings.create.assert_awaited_once()

    @patch("src.llm.embeddings.get_settings")
    @patch("src.llm.embeddings.openai.AsyncOpenAI")
    async def test_embed_caching_prevents_duplicate_api_call(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """Calling embed() twice with the same text should hit the API only once."""
        mock_get_settings.return_value = _make_settings()

        expected_vector = [0.5] * 1536
        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(
            return_value=_make_embedding_response([expected_vector])
        )
        mock_openai_cls.return_value = mock_client

        service = EmbeddingService()

        first = await service.embed("동일 텍스트")
        second = await service.embed("동일 텍스트")

        assert first == second
        # API should be called exactly once; second call served from cache
        assert mock_client.embeddings.create.await_count == 1

    @patch("src.llm.embeddings.get_settings")
    @patch("src.llm.embeddings.openai.AsyncOpenAI")
    async def test_embed_batch_uses_cache_for_known_texts(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """embed_batch() should skip API calls for texts already in the cache."""
        mock_get_settings.return_value = _make_settings()

        vec_cached = [0.1] * 1536
        vec_new = [0.9] * 1536

        mock_client = MagicMock()
        # First call: embed the single text to populate cache
        mock_client.embeddings.create = AsyncMock(
            side_effect=[
                _make_embedding_response([vec_cached]),
                _make_embedding_response([vec_new]),
            ]
        )
        mock_openai_cls.return_value = mock_client

        service = EmbeddingService()

        # Pre-populate cache
        await service.embed("cached-text")

        # Batch that includes already-cached text
        results = await service.embed_batch(["cached-text", "new-text"])

        assert len(results) == 2
        assert results[0] == vec_cached
        assert results[1] == vec_new
        # Two API calls total: one for embed(), one for the uncached "new-text" in embed_batch()
        assert mock_client.embeddings.create.await_count == 2

    @patch("src.llm.embeddings.get_settings")
    @patch("src.llm.embeddings.openai.AsyncOpenAI")
    async def test_embed_batch_multiple_batches(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """embed_batch() should split into batches respecting batch_size."""
        mock_get_settings.return_value = _make_settings()

        texts = [f"text-{i}" for i in range(5)]
        vecs = [[float(i)] * 1536 for i in range(5)]

        mock_client = MagicMock()
        # batch_size=3 means: first batch 3 texts, second batch 2 texts
        mock_client.embeddings.create = AsyncMock(
            side_effect=[
                _make_embedding_response(vecs[:3]),
                _make_embedding_response(vecs[3:]),
            ]
        )
        mock_openai_cls.return_value = mock_client

        service = EmbeddingService()
        results = await service.embed_batch(texts, batch_size=3)

        assert len(results) == 5
        assert results[0] == vecs[0]
        assert results[4] == vecs[4]
        assert mock_client.embeddings.create.await_count == 2

    @patch("src.llm.embeddings.get_settings")
    @patch("src.llm.embeddings.openai.AsyncOpenAI")
    async def test_embed_different_texts_cached_independently(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """Different texts should each get their own cache entry."""
        mock_get_settings.return_value = _make_settings()

        vec_a = [0.1] * 1536
        vec_b = [0.2] * 1536

        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock(
            side_effect=[
                _make_embedding_response([vec_a]),
                _make_embedding_response([vec_b]),
            ]
        )
        mock_openai_cls.return_value = mock_client

        service = EmbeddingService()

        result_a = await service.embed("text-a")
        result_b = await service.embed("text-b")

        assert result_a == vec_a
        assert result_b == vec_b
        assert mock_client.embeddings.create.await_count == 2

    @patch("src.llm.embeddings.get_settings")
    @patch("src.llm.embeddings.openai.AsyncOpenAI")
    async def test_embed_batch_empty_list(
        self,
        mock_openai_cls: MagicMock,
        mock_get_settings: MagicMock,
    ) -> None:
        """embed_batch() with an empty list should return empty without calling API."""
        mock_get_settings.return_value = _make_settings()

        mock_client = MagicMock()
        mock_client.embeddings.create = AsyncMock()
        mock_openai_cls.return_value = mock_client

        service = EmbeddingService()
        results = await service.embed_batch([])

        assert results == []
        mock_client.embeddings.create.assert_not_awaited()
