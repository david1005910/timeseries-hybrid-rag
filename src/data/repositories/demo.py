"""In-memory demo repositories for testing without external services.

Provides sample data for the infrastructure incident scenario:
- CPU spike caused by batch-job-7842
- Memory pressure, connection pool exhaustion
- Full causal chain from traffic spike to service degradation

Activated by DEMO_MODE=true in .env
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta, timezone
from typing import Any, AsyncIterator

from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Sample Data
# ---------------------------------------------------------------------------

SAMPLE_TIMESERIES = [
    {"time": "2025-01-31T14:25:00Z", "measurement": "system_metrics", "field": "cpu_usage", "value": 32.5, "tags": {"host": "server-01", "service": "api-gateway", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:28:00Z", "measurement": "system_metrics", "field": "cpu_usage", "value": 35.2, "tags": {"host": "server-01", "service": "api-gateway", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:30:00Z", "measurement": "system_metrics", "field": "cpu_usage", "value": 72.8, "tags": {"host": "server-01", "service": "data-service", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:31:00Z", "measurement": "system_metrics", "field": "cpu_usage", "value": 88.3, "tags": {"host": "server-01", "service": "data-service", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:32:00Z", "measurement": "system_metrics", "field": "cpu_usage", "value": 95.1, "tags": {"host": "server-01", "service": "data-service", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:33:00Z", "measurement": "system_metrics", "field": "cpu_usage", "value": 91.7, "tags": {"host": "server-01", "service": "api-gateway", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:35:00Z", "measurement": "system_metrics", "field": "cpu_usage", "value": 78.4, "tags": {"host": "server-01", "service": "api-gateway", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:40:00Z", "measurement": "system_metrics", "field": "cpu_usage", "value": 45.0, "tags": {"host": "server-01", "service": "api-gateway", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:30:00Z", "measurement": "system_metrics", "field": "memory_usage", "value": 70.2, "tags": {"host": "server-01", "service": "data-service", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:31:00Z", "measurement": "system_metrics", "field": "memory_usage", "value": 78.5, "tags": {"host": "server-01", "service": "data-service", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:32:00Z", "measurement": "system_metrics", "field": "memory_usage", "value": 88.5, "tags": {"host": "server-01", "service": "data-service", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:35:00Z", "measurement": "system_metrics", "field": "memory_usage", "value": 75.0, "tags": {"host": "server-01", "service": "data-service", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:40:00Z", "measurement": "system_metrics", "field": "memory_usage", "value": 60.1, "tags": {"host": "server-01", "service": "api-gateway", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:29:00Z", "measurement": "system_metrics", "field": "connection_pool_usage", "value": 48.0, "tags": {"host": "server-01", "service": "data-service", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:30:00Z", "measurement": "system_metrics", "field": "connection_pool_usage", "value": 50.0, "tags": {"host": "server-01", "service": "data-service", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:25:00Z", "measurement": "system_metrics", "field": "error_rate", "value": 0.1, "tags": {"host": "server-01", "service": "api-gateway", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:33:00Z", "measurement": "system_metrics", "field": "error_rate", "value": 12.5, "tags": {"host": "server-01", "service": "api-gateway", "region": "ap-northeast-2"}},
    {"time": "2025-01-31T14:35:00Z", "measurement": "system_metrics", "field": "error_rate", "value": 5.3, "tags": {"host": "server-01", "service": "api-gateway", "region": "ap-northeast-2"}},
]

SAMPLE_DOCUMENTS = [
    {
        "id": "doc-001",
        "content": "CPU 사용률이 95%까지 급증한 것은 대규모 배치 작업(Job ID: batch-7842)이 14:30에 시작되면서 발생했습니다. 이 배치 작업은 데이터 집계를 수행하며, 평소보다 3배 많은 데이터를 처리했습니다.",
        "source": "incident-report-2025-01-31",
        "source_type": "vector",
        "relevance_score": 0.95,
        "metadata": {"type": "incident_report"},
    },
    {
        "id": "doc-002",
        "content": "메모리 사용량이 70%에서 88%로 증가한 원인은 배치 작업이 대량의 임시 데이터를 메모리에 적재했기 때문입니다. GC 압력이 증가하면서 추가적인 CPU 부하가 발생했습니다.",
        "source": "incident-report-2025-01-31",
        "source_type": "vector",
        "relevance_score": 0.91,
        "metadata": {"type": "incident_report"},
    },
    {
        "id": "doc-003",
        "content": "데이터베이스 연결 풀 고갈은 트래픽이 평소 대비 300% 증가한 시점에서 발생했습니다. max_connections=50 설정이 증가된 트래픽을 감당하지 못했으며, 연결 대기 시간이 30초를 초과했습니다.",
        "source": "postmortem-2025-01-31",
        "source_type": "vector",
        "relevance_score": 0.87,
        "metadata": {"type": "postmortem"},
    },
    {
        "id": "doc-004",
        "content": "API 게이트웨이 응답 지연이 5초를 초과하면서 로드밸런서 헬스체크가 실패했습니다. 이로 인해 서비스가 일시적으로 다운되었으며, 약 15분간 사용자에게 503 에러가 반환되었습니다.",
        "source": "postmortem-2025-01-31",
        "source_type": "vector",
        "relevance_score": 0.82,
        "metadata": {"type": "postmortem"},
    },
    {
        "id": "doc-005",
        "content": "권장 조치: 1) 데이터베이스 연결 풀 크기를 150으로 증가 2) 배치 작업 스케줄링 개선 (피크 시간 회피) 3) Auto-scaling 임계값 조정 4) 메모리 할당 최적화",
        "source": "postmortem-2025-01-31",
        "source_type": "vector",
        "relevance_score": 0.75,
        "metadata": {"type": "recommendation"},
    },
    {
        "id": "doc-006",
        "content": "시계열 데이터 분석 결과, 매주 화요일 14:00-16:00에 트래픽이 급증하는 패턴이 발견되었습니다. 이는 정기 배치 작업과 사용자 활동 피크가 겹치는 시간대입니다.",
        "source": "analysis-report-weekly",
        "source_type": "vector",
        "relevance_score": 0.70,
        "metadata": {"type": "analysis"},
    },
    {
        "id": "doc-007",
        "content": "서버 클러스터 구성: cluster-1은 server-01(8코어), server-02(16코어), server-03(8코어)로 구성됩니다. 로드밸런서는 라운드로빈 방식으로 요청을 분배합니다.",
        "source": "infrastructure-docs",
        "source_type": "vector",
        "relevance_score": 0.65,
        "metadata": {"type": "documentation"},
    },
    {
        "id": "doc-008",
        "content": "프로메테우스 알림 규칙: CPU > 80% (5분) → WARNING, CPU > 90% (2분) → CRITICAL, Memory > 85% (3분) → WARNING, Error rate > 5% (1분) → CRITICAL",
        "source": "alerting-rules",
        "source_type": "vector",
        "relevance_score": 0.60,
        "metadata": {"type": "configuration"},
    },
]

SAMPLE_ENTITIES = [
    {"id": "e1", "name": "server-01", "type": "entity", "labels": ["entity"], "properties": {"type": "server", "status": "running", "cpu_cores": 8}},
    {"id": "e2", "name": "server-02", "type": "entity", "labels": ["entity"], "properties": {"type": "server", "status": "running", "cpu_cores": 16}},
    {"id": "e3", "name": "server-03", "type": "entity", "labels": ["entity"], "properties": {"type": "server", "status": "degraded", "cpu_cores": 8}},
    {"id": "e4", "name": "api-gateway", "type": "entity", "labels": ["entity"], "properties": {"type": "service", "version": "2.1.0"}},
    {"id": "e5", "name": "auth-service", "type": "entity", "labels": ["entity"], "properties": {"type": "service", "version": "1.5.0"}},
    {"id": "e6", "name": "data-service", "type": "entity", "labels": ["entity"], "properties": {"type": "service", "version": "3.0.0"}},
    {"id": "e7", "name": "cluster-1", "type": "entity", "labels": ["entity"], "properties": {"type": "cluster", "region": "ap-northeast-2"}},
    {"id": "e8", "name": "load-balancer", "type": "entity", "labels": ["entity"], "properties": {"type": "infrastructure", "algorithm": "round-robin"}},
    {"id": "e9", "name": "database-pool", "type": "entity", "labels": ["entity"], "properties": {"type": "infrastructure", "max_connections": 50}},
    {"id": "m1", "name": "cpu_usage", "type": "metric", "labels": ["metric"], "properties": {"unit": "%", "source": "prometheus"}},
    {"id": "m2", "name": "memory_usage", "type": "metric", "labels": ["metric"], "properties": {"unit": "%", "source": "prometheus"}},
    {"id": "m3", "name": "request_latency", "type": "metric", "labels": ["metric"], "properties": {"unit": "ms", "source": "prometheus"}},
    {"id": "m4", "name": "error_rate", "type": "metric", "labels": ["metric"], "properties": {"unit": "%", "source": "prometheus"}},
    {"id": "m5", "name": "connection_pool_usage", "type": "metric", "labels": ["metric"], "properties": {"unit": "count", "source": "prometheus"}},
    {"id": "ev1", "name": "batch-job-7842", "type": "event", "labels": ["event"], "properties": {"severity": "info", "timestamp": "2025-01-31T14:30:00"}},
    {"id": "ev2", "name": "cpu-spike", "type": "event", "labels": ["event"], "properties": {"severity": "critical", "timestamp": "2025-01-31T14:32:00"}},
    {"id": "ev3", "name": "memory-pressure", "type": "event", "labels": ["event"], "properties": {"severity": "warning", "timestamp": "2025-01-31T14:31:00"}},
    {"id": "ev4", "name": "service-degradation", "type": "event", "labels": ["event"], "properties": {"severity": "critical", "timestamp": "2025-01-31T14:33:00"}},
    {"id": "ev5", "name": "connection-pool-exhaustion", "type": "event", "labels": ["event"], "properties": {"severity": "critical", "timestamp": "2025-01-31T14:29:00"}},
    {"id": "ev6", "name": "traffic-spike-300pct", "type": "event", "labels": ["event"], "properties": {"severity": "warning", "timestamp": "2025-01-31T14:25:00"}},
    {"id": "c1", "name": "resource-contention", "type": "concept", "labels": ["concept"], "properties": {"description": "리소스 경합으로 인한 성능 저하"}},
    {"id": "c2", "name": "cascading-failure", "type": "concept", "labels": ["concept"], "properties": {"description": "연쇄 장애 패턴"}},
    {"id": "c3", "name": "capacity-planning", "type": "concept", "labels": ["concept"], "properties": {"description": "용량 계획 필요"}},
]

SAMPLE_GRAPH_PATHS = [
    {
        "nodes": [
            {"id": "ev6", "name": "traffic-spike-300pct", "labels": ["event"]},
            {"id": "ev5", "name": "connection-pool-exhaustion", "labels": ["event"]},
            {"id": "ev1", "name": "batch-job-7842", "labels": ["event"]},
            {"id": "ev3", "name": "memory-pressure", "labels": ["event"]},
            {"id": "ev2", "name": "cpu-spike", "labels": ["event"]},
            {"id": "ev4", "name": "service-degradation", "labels": ["event"]},
        ],
        "relationships": [
            {"type": "CAUSES", "confidence": 0.85},
            {"type": "CAUSES", "confidence": 0.75},
            {"type": "CAUSES", "confidence": 0.90},
            {"type": "CAUSES", "confidence": 0.88},
            {"type": "CAUSES", "confidence": 0.92},
        ],
        "hops": 5,
    },
    {
        "nodes": [
            {"id": "ev1", "name": "batch-job-7842", "labels": ["event"]},
            {"id": "ev3", "name": "memory-pressure", "labels": ["event"]},
            {"id": "ev2", "name": "cpu-spike", "labels": ["event"]},
        ],
        "relationships": [
            {"type": "CAUSES", "confidence": 0.90},
            {"type": "CAUSES", "confidence": 0.88},
        ],
        "hops": 2,
    },
    {
        "nodes": [
            {"id": "m1", "name": "cpu_usage", "labels": ["metric"]},
            {"id": "m2", "name": "memory_usage", "labels": ["metric"]},
        ],
        "relationships": [
            {"type": "CORRELATES", "confidence": 0.78},
        ],
        "hops": 1,
    },
]

_ENTITY_BY_ID = {e["id"]: e for e in SAMPLE_ENTITIES}


# ---------------------------------------------------------------------------
# Demo Timeseries Repository
# ---------------------------------------------------------------------------

class DemoTimeseriesRepository:
    """인메모리 시계열 데이터 리포지토리."""

    async def query_metrics(
        self,
        measurement: str,
        tags: dict[str, str] | None = None,
        fields: list[str] | None = None,
        start: str = "-1h",
        stop: str = "now()",
        aggregation: str | None = None,
        group_by: str | None = None,
    ) -> list[dict[str, Any]]:
        records = SAMPLE_TIMESERIES
        if fields:
            records = [r for r in records if r["field"] in fields]
        if tags:
            records = [
                r for r in records
                if all(r["tags"].get(k) == v for k, v in tags.items())
            ]
        logger.info("demo_timeseries_query", measurement=measurement, record_count=len(records))
        return records

    async def write_metrics(self, measurement: str, tags: dict[str, str], fields: dict[str, float]) -> None:
        logger.info("demo_timeseries_write", measurement=measurement)

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Demo Vector Repository
# ---------------------------------------------------------------------------

class DemoVectorRepository:
    """인메모리 벡터 검색 리포지토리."""

    async def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        source_type: str | None = None,
    ) -> list[dict[str, Any]]:
        docs = SAMPLE_DOCUMENTS
        if source_type:
            docs = [d for d in docs if d["source_type"] == source_type]
        results = [
            {
                "id": doc["id"],
                "score": doc["relevance_score"],
                "content": doc["content"],
                "source": doc["source"],
                "source_type": doc["source_type"],
                "metadata_json": json.dumps(doc["metadata"], ensure_ascii=False),
            }
            for doc in docs[:top_k]
        ]
        logger.info("demo_vector_search", hits=len(results))
        return results

    async def insert(self, ids: list[str], embeddings: list[list[float]], contents: list[str],
                     sources: list[str], source_types: list[str], metadata_jsons: list[str]) -> int:
        return len(ids)

    async def delete(self, ids: list[str]) -> None:
        pass

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Demo Graph Repository
# ---------------------------------------------------------------------------

class _DemoAsyncResult:
    """Neo4j async result를 모방하는 비동기 이터레이터."""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self._records = records
        self._index = 0

    def __aiter__(self) -> _DemoAsyncResult:
        self._index = 0
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._index >= len(self._records):
            raise StopAsyncIteration
        record = self._records[self._index]
        self._index += 1
        return record

    async def single(self) -> dict[str, Any] | None:
        return self._records[0] if self._records else None


class _DemoSession:
    """Neo4j 세션을 모방하는 데모 세션."""

    async def run(self, query: str, **kwargs: Any) -> _DemoAsyncResult:
        # list_entities 쿼리 처리
        if "RETURN n.id as id" in query:
            limit = kwargs.get("limit", 50)
            node_type = None
            for entity in SAMPLE_ENTITIES:
                if entity["type"] in query:
                    node_type = entity["type"]
                    break

            entities = SAMPLE_ENTITIES
            if node_type:
                entities = [e for e in entities if e["type"] == node_type]

            records = [
                {"id": e["id"], "name": e["name"], "labels": e["labels"]}
                for e in entities[:limit]
            ]
            return _DemoAsyncResult(records)

        # get_node 쿼리 처리
        if "MATCH (n {id: $id})" in query:
            node_id = kwargs.get("id")
            entity = _ENTITY_BY_ID.get(node_id)
            if entity:
                node_data = {"id": entity["id"], "name": entity["name"], **entity.get("properties", {})}
                return _DemoAsyncResult([{"n": node_data, "labels": entity["labels"]}])
            return _DemoAsyncResult([])

        # traverse 쿼리 처리
        if "MATCH path" in query:
            start_id = kwargs.get("start_id")
            min_confidence = kwargs.get("min_confidence", 0.0)
            paths = []
            for path in SAMPLE_GRAPH_PATHS:
                nodes = path["nodes"]
                if any(n["id"] == start_id for n in nodes):
                    if all(r["confidence"] >= min_confidence for r in path["relationships"]):
                        paths.append({
                            "nodes": nodes,
                            "rels": path["relationships"],
                            "hops": path["hops"],
                        })
            return _DemoAsyncResult(paths)

        return _DemoAsyncResult([])

    async def close(self) -> None:
        pass


class _DemoSessionCtx:
    """async with 컨텍스트 매니저."""

    async def __aenter__(self) -> _DemoSession:
        return _DemoSession()

    async def __aexit__(self, *args: Any) -> None:
        pass


class DemoGraphRepository:
    """인메모리 지식 그래프 리포지토리."""

    async def _get_session(self) -> _DemoSessionCtx:
        return _DemoSessionCtx()

    async def get_node(self, node_id: str) -> dict[str, Any] | None:
        entity = _ENTITY_BY_ID.get(node_id)
        if entity:
            node_data = {"id": entity["id"], "name": entity["name"], **entity.get("properties", {})}
            node_data["type"] = entity["type"]
            return node_data
        return None

    async def traverse(
        self,
        start_node_id: str,
        max_hops: int = 5,
        relationship_types: list[str] | None = None,
        min_confidence: float = 0.0,
    ) -> list[dict[str, Any]]:
        results = []
        for path in SAMPLE_GRAPH_PATHS:
            nodes = path["nodes"]
            if not any(n["id"] == start_node_id for n in nodes):
                continue
            if path["hops"] > max_hops:
                continue
            if not all(r["confidence"] >= min_confidence for r in path["relationships"]):
                continue
            if relationship_types:
                types_upper = {t.upper() for t in relationship_types}
                if not all(r["type"] in types_upper for r in path["relationships"]):
                    continue
            results.append({
                "nodes": nodes,
                "relationships": path["relationships"],
                "hops": path["hops"],
            })
        logger.info("demo_graph_traverse", start=start_node_id, paths_found=len(results))
        return results

    async def find_causal_chain(
        self,
        start_node_id: str,
        end_node_id: str | None = None,
        max_hops: int = 5,
    ) -> list[dict[str, Any]]:
        return await self.traverse(
            start_node_id=start_node_id,
            max_hops=max_hops,
            relationship_types=["causes"],
            min_confidence=0.5,
        )

    async def create_node(self, node: Any) -> Any:
        return node

    async def create_relationship(self, rel: Any) -> Any:
        return rel

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Demo Embedding Service
# ---------------------------------------------------------------------------

class DemoEmbeddingService:
    """더미 임베딩 서비스 (벡터 검색이 내용 기반이므로 실제 임베딩 불필요)."""

    async def embed(self, text: str) -> list[float]:
        return [0.1] * 1536

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] * 1536 for _ in texts]
