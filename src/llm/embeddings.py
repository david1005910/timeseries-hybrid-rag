"""Embedding generation with caching."""
from __future__ import annotations

from typing import Any

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """OpenAI text-embedding-3-large 임베딩 생성기."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        self._dim = settings.embedding_dimension
        self._cache: dict[str, list[float]] = {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=30))
    async def embed(self, text: str) -> list[float]:
        """단일 텍스트 임베딩 생성."""
        if text in self._cache:
            return self._cache[text]

        response = await self._client.embeddings.create(
            model=self._model,
            input=text,
            dimensions=self._dim,
        )
        embedding = response.data[0].embedding
        self._cache[text] = embedding
        logger.info("embedding_generated", model=self._model, text_len=len(text))
        return embedding

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=30))
    async def embed_batch(self, texts: list[str], batch_size: int = 100) -> list[list[float]]:
        """배치 임베딩 생성."""
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Check cache first
            uncached_texts = [t for t in batch if t not in self._cache]
            if uncached_texts:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=uncached_texts,
                    dimensions=self._dim,
                )
                for text_item, data in zip(uncached_texts, response.data):
                    self._cache[text_item] = data.embedding

            all_embeddings.extend([self._cache[t] for t in batch])

        logger.info("batch_embedding_generated", model=self._model, count=len(texts))
        return all_embeddings
