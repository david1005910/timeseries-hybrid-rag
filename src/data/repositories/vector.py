"""Milvus Repository for vector search operations."""
from __future__ import annotations

import time
from typing import Any

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility

from src.config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

COLLECTION_NAME = "documents"
EMBEDDING_DIM = 1536


class VectorRepository:
    """벡터 유사도 검색을 위한 Milvus Repository."""

    def __init__(self) -> None:
        settings = get_settings()
        self._host = settings.milvus_host
        self._port = settings.milvus_port
        self._dim = settings.embedding_dimension
        self._connected = False
        self._collection: Collection | None = None

    def _connect(self) -> None:
        if not self._connected:
            connections.connect("default", host=self._host, port=self._port)
            self._connected = True

    def _ensure_collection(self) -> Collection:
        self._connect()
        if self._collection is not None:
            return self._collection

        if utility.has_collection(COLLECTION_NAME):
            self._collection = Collection(COLLECTION_NAME)
        else:
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=128),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self._dim),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="source_type", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="metadata_json", dtype=DataType.VARCHAR, max_length=40960),
            ]
            schema = CollectionSchema(fields, description="Document embeddings for RAG")
            self._collection = Collection(COLLECTION_NAME, schema)
            self._collection.create_index(
                field_name="embedding",
                index_params={"index_type": "IVF_FLAT", "metric_type": "IP", "params": {"nlist": 128}},
            )
        self._collection.load()
        return self._collection

    async def insert(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        contents: list[str],
        sources: list[str],
        source_types: list[str],
        metadata_jsons: list[str],
    ) -> int:
        """벡터 데이터 삽입.

        Returns:
            삽입된 레코드 수
        """
        collection = self._ensure_collection()
        data = [ids, embeddings, contents, sources, source_types, metadata_jsons]
        result = collection.insert(data)
        collection.flush()
        count = result.insert_count
        logger.info("vector_insert", count=count)
        return count

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        source_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """벡터 유사도 검색.

        Args:
            query_embedding: 질의 벡터
            top_k: 반환할 상위 K개 결과
            source_type: 소스 유형 필터

        Returns:
            유사도 검색 결과
        """
        t0 = time.time()
        collection = self._ensure_collection()

        search_params = {"metric_type": "IP", "params": {"nprobe": 10}}
        expr = f'source_type == "{source_type}"' if source_type else None

        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["content", "source", "source_type", "metadata_json"],
        )

        hits: list[dict[str, Any]] = []
        for hit in results[0]:
            hits.append({
                "id": hit.id,
                "score": hit.score,
                "content": hit.entity.get("content"),
                "source": hit.entity.get("source"),
                "source_type": hit.entity.get("source_type"),
                "metadata_json": hit.entity.get("metadata_json"),
            })

        elapsed = (time.time() - t0) * 1000
        logger.info("vector_search", top_k=top_k, hits=len(hits), elapsed_ms=round(elapsed, 2))
        return hits

    async def delete(self, ids: list[str]) -> None:
        """벡터 삭제."""
        collection = self._ensure_collection()
        expr = f'id in {ids}'
        collection.delete(expr)
        logger.info("vector_delete", count=len(ids))

    def close(self) -> None:
        if self._connected:
            connections.disconnect("default")
            self._connected = False
