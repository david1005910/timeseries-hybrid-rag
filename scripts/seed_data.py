"""Seed data script for development and testing.

Populates all databases with sample data:
- InfluxDB: System metrics (CPU, memory, disk, network)
- Neo4j: Knowledge graph (servers, services, events, metrics)
- Milvus: Document embeddings (via OpenAI)
- PostgreSQL: Test user

Usage:
    poetry run python scripts/seed_data.py
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import get_settings
from src.utils.logging import setup_logging, get_logger


# --- Sample Time-Series Data ---
async def seed_influxdb() -> None:
    """시계열 메트릭 샘플 데이터 생성."""
    logger = get_logger("seed")
    settings = get_settings()
    try:
        from influxdb_client import InfluxDBClient, Point
        from influxdb_client.client.write_api import SYNCHRONOUS

        client = InfluxDBClient(
            url=settings.influxdb_url,
            token=settings.influxdb_token,
            org=settings.influxdb_org,
        )
        write_api = client.write_api(write_options=SYNCHRONOUS)

        hosts = ["server-01", "server-02", "server-03"]
        services = ["api-gateway", "auth-service", "data-service"]
        now = datetime.utcnow()

        points = []
        for i in range(1000):
            ts = now - timedelta(minutes=i)
            host = random.choice(hosts)
            service = random.choice(services)

            # Normal CPU with spike at ~500 minutes ago
            cpu_base = 30 + random.gauss(0, 5)
            if 490 <= i <= 510:
                cpu_base = 85 + random.gauss(0, 5)  # Spike

            memory_base = 55 + random.gauss(0, 3)
            if 490 <= i <= 520:
                memory_base = 80 + random.gauss(0, 3)

            points.append(
                Point("system_metrics")
                .tag("host", host)
                .tag("service", service)
                .tag("region", "ap-northeast-2")
                .field("cpu_usage", max(0, min(100, cpu_base)))
                .field("memory_usage", max(0, min(100, memory_base)))
                .field("disk_io", max(0, random.gauss(50, 15)))
                .field("network_throughput", max(0, random.gauss(200, 50)))
                .time(ts)
            )

        write_api.write(bucket=settings.influxdb_bucket, org=settings.influxdb_org, record=points)
        client.close()
        logger.info("influxdb_seeded", points=len(points))
    except Exception as e:
        logger.error("influxdb_seed_failed", error=str(e))


# --- Sample Knowledge Graph ---
async def seed_neo4j() -> None:
    """지식 그래프 샘플 데이터 생성."""
    logger = get_logger("seed")
    settings = get_settings()
    try:
        from neo4j import AsyncGraphDatabase

        driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

        async with driver.session() as session:
            # Clear existing data
            await session.run("MATCH (n) DETACH DELETE n")

            # Create entities (servers, services, clusters)
            entities = [
                ("entity", "e1", "server-01", {"type": "server", "status": "running", "cpu_cores": 8}),
                ("entity", "e2", "server-02", {"type": "server", "status": "running", "cpu_cores": 16}),
                ("entity", "e3", "server-03", {"type": "server", "status": "degraded", "cpu_cores": 8}),
                ("entity", "e4", "api-gateway", {"type": "service", "version": "2.1.0"}),
                ("entity", "e5", "auth-service", {"type": "service", "version": "1.5.0"}),
                ("entity", "e6", "data-service", {"type": "service", "version": "3.0.0"}),
                ("entity", "e7", "cluster-1", {"type": "cluster", "region": "ap-northeast-2"}),
                ("entity", "e8", "load-balancer", {"type": "infrastructure", "algorithm": "round-robin"}),
                ("entity", "e9", "database-pool", {"type": "infrastructure", "max_connections": 50}),
            ]

            for node_type, node_id, name, props in entities:
                props_str = ", ".join(f"{k}: ${k}" for k in props)
                query = f"CREATE (n:{node_type} {{id: $id, name: $name, {props_str}}})"
                await session.run(query, id=node_id, name=name, **props)

            # Create metrics
            metrics = [
                ("metric", "m1", "cpu_usage", {"unit": "%", "source": "prometheus"}),
                ("metric", "m2", "memory_usage", {"unit": "%", "source": "prometheus"}),
                ("metric", "m3", "request_latency", {"unit": "ms", "source": "prometheus"}),
                ("metric", "m4", "error_rate", {"unit": "%", "source": "prometheus"}),
                ("metric", "m5", "connection_pool_usage", {"unit": "count", "source": "prometheus"}),
            ]

            for node_type, node_id, name, props in metrics:
                props_str = ", ".join(f"{k}: ${k}" for k in props)
                query = f"CREATE (n:{node_type} {{id: $id, name: $name, {props_str}}})"
                await session.run(query, id=node_id, name=name, **props)

            # Create events
            events = [
                ("event", "ev1", "batch-job-7842", {"severity": "info", "timestamp": "2025-01-31T14:30:00"}),
                ("event", "ev2", "cpu-spike", {"severity": "critical", "timestamp": "2025-01-31T14:32:00"}),
                ("event", "ev3", "memory-pressure", {"severity": "warning", "timestamp": "2025-01-31T14:31:00"}),
                ("event", "ev4", "service-degradation", {"severity": "critical", "timestamp": "2025-01-31T14:33:00"}),
                ("event", "ev5", "connection-pool-exhaustion", {"severity": "critical", "timestamp": "2025-01-31T14:29:00"}),
                ("event", "ev6", "traffic-spike-300pct", {"severity": "warning", "timestamp": "2025-01-31T14:25:00"}),
            ]

            for node_type, node_id, name, props in events:
                props_str = ", ".join(f"{k}: ${k}" for k in props)
                query = f"CREATE (n:{node_type} {{id: $id, name: $name, {props_str}}})"
                await session.run(query, id=node_id, name=name, **props)

            # Create concepts
            concepts = [
                ("concept", "c1", "resource-contention", {"description": "리소스 경합으로 인한 성능 저하"}),
                ("concept", "c2", "cascading-failure", {"description": "연쇄 장애 패턴"}),
                ("concept", "c3", "capacity-planning", {"description": "용량 계획 필요"}),
            ]

            for node_type, node_id, name, props in concepts:
                props_str = ", ".join(f"{k}: ${k}" for k in props)
                query = f"CREATE (n:{node_type} {{id: $id, name: $name, {props_str}}})"
                await session.run(query, id=node_id, name=name, **props)

            # Create relationships (causal chain for incident analysis)
            relationships = [
                # Cluster structure
                ("e1", "e7", "BELONGS_TO", 1.0),
                ("e2", "e7", "BELONGS_TO", 1.0),
                ("e3", "e7", "BELONGS_TO", 1.0),
                ("e4", "e1", "BELONGS_TO", 0.9),
                ("e5", "e2", "BELONGS_TO", 0.9),
                ("e6", "e3", "BELONGS_TO", 0.9),
                # Causal chain: Traffic spike → Connection pool exhaustion → CPU spike → Service degradation
                ("ev6", "ev5", "CAUSES", 0.85),      # Traffic → Connection pool
                ("ev5", "ev1", "CAUSES", 0.75),       # Connection pool → Batch job start
                ("ev1", "ev3", "CAUSES", 0.90),       # Batch job → Memory pressure
                ("ev3", "ev2", "CAUSES", 0.88),       # Memory pressure → CPU spike
                ("ev2", "ev4", "CAUSES", 0.92),       # CPU spike → Service degradation
                # Metric correlations
                ("m1", "m2", "CORRELATES", 0.78),     # CPU ↔ Memory
                ("m5", "m3", "CORRELATES", 0.82),     # Connection pool → Latency
                ("m4", "ev4", "CORRELATES", 0.90),    # Error rate ↔ Service degradation
                # Temporal precedence
                ("ev6", "ev5", "PRECEDES", 1.0),
                ("ev5", "ev1", "PRECEDES", 1.0),
                ("ev1", "ev3", "PRECEDES", 1.0),
                ("ev3", "ev2", "PRECEDES", 1.0),
                # Concept links
                ("ev5", "c1", "CORRELATES", 0.85),
                ("ev4", "c2", "CORRELATES", 0.90),
                ("ev6", "c3", "CORRELATES", 0.80),
            ]

            for src, tgt, rel_type, confidence in relationships:
                await session.run(
                    f"""
                    MATCH (a {{id: $src}}), (b {{id: $tgt}})
                    CREATE (a)-[r:{rel_type} {{confidence: $confidence, created_at: datetime()}}]->(b)
                    """,
                    src=src, tgt=tgt, confidence=confidence,
                )

        await driver.close()
        logger.info("neo4j_seeded", entities=len(entities), metrics=len(metrics), events=len(events), relationships=len(relationships))
    except Exception as e:
        logger.error("neo4j_seed_failed", error=str(e))


# --- Sample Documents for Vector Search ---
SAMPLE_DOCUMENTS = [
    {
        "id": "doc-001",
        "content": "CPU 사용률이 95%까지 급증한 것은 대규모 배치 작업(Job ID: batch-7842)이 14:30에 시작되면서 발생했습니다. 이 배치 작업은 데이터 집계를 수행하며, 평소보다 3배 많은 데이터를 처리했습니다.",
        "source": "incident-report-2025-01-31",
        "source_type": "document",
    },
    {
        "id": "doc-002",
        "content": "메모리 사용량이 70%에서 88%로 증가한 원인은 배치 작업이 대량의 임시 데이터를 메모리에 적재했기 때문입니다. GC 압력이 증가하면서 추가적인 CPU 부하가 발생했습니다.",
        "source": "incident-report-2025-01-31",
        "source_type": "document",
    },
    {
        "id": "doc-003",
        "content": "데이터베이스 연결 풀 고갈은 트래픽이 평소 대비 300% 증가한 시점에서 발생했습니다. max_connections=50 설정이 증가된 트래픽을 감당하지 못했으며, 연결 대기 시간이 30초를 초과했습니다.",
        "source": "postmortem-2025-01-31",
        "source_type": "document",
    },
    {
        "id": "doc-004",
        "content": "API 게이트웨이 응답 지연이 5초를 초과하면서 로드밸런서 헬스체크가 실패했습니다. 이로 인해 서비스가 일시적으로 다운되었으며, 약 15분간 사용자에게 503 에러가 반환되었습니다.",
        "source": "postmortem-2025-01-31",
        "source_type": "document",
    },
    {
        "id": "doc-005",
        "content": "권장 조치: 1) 데이터베이스 연결 풀 크기를 150으로 증가 2) 배치 작업 스케줄링 개선 (피크 시간 회피) 3) Auto-scaling 임계값 조정 4) 메모리 할당 최적화",
        "source": "postmortem-2025-01-31",
        "source_type": "document",
    },
    {
        "id": "doc-006",
        "content": "시계열 데이터 분석 결과, 매주 화요일 14:00-16:00에 트래픽이 급증하는 패턴이 발견되었습니다. 이는 정기 배치 작업과 사용자 활동 피크가 겹치는 시간대입니다.",
        "source": "analysis-report-weekly",
        "source_type": "document",
    },
    {
        "id": "doc-007",
        "content": "서버 클러스터 구성: cluster-1은 server-01(8코어), server-02(16코어), server-03(8코어)로 구성됩니다. 로드밸런서는 라운드로빈 방식으로 요청을 분배합니다.",
        "source": "infrastructure-docs",
        "source_type": "document",
    },
    {
        "id": "doc-008",
        "content": "프로메테우스 알림 규칙: CPU > 80% (5분) → WARNING, CPU > 90% (2분) → CRITICAL, Memory > 85% (3분) → WARNING, Error rate > 5% (1분) → CRITICAL",
        "source": "alerting-rules",
        "source_type": "document",
    },
]


async def seed_milvus() -> None:
    """벡터 저장소 샘플 데이터 생성."""
    logger = get_logger("seed")
    settings = get_settings()

    if not settings.openai_api_key or settings.openai_api_key == "your-openai-api-key":
        logger.warning("openai_api_key_not_set", message="Skipping Milvus seeding (no OpenAI API key)")
        return

    try:
        from src.llm.embeddings import EmbeddingService
        from src.data.repositories.vector import VectorRepository

        embedding_service = EmbeddingService()
        vector_repo = VectorRepository()

        texts = [doc["content"] for doc in SAMPLE_DOCUMENTS]
        embeddings = await embedding_service.embed_batch(texts)

        ids = [doc["id"] for doc in SAMPLE_DOCUMENTS]
        contents = texts
        sources = [doc["source"] for doc in SAMPLE_DOCUMENTS]
        source_types = [doc["source_type"] for doc in SAMPLE_DOCUMENTS]
        metadata_jsons = [json.dumps({"source": doc["source"]}, ensure_ascii=False) for doc in SAMPLE_DOCUMENTS]

        count = await vector_repo.insert(ids, embeddings, contents, sources, source_types, metadata_jsons)
        vector_repo.close()
        logger.info("milvus_seeded", documents=count)
    except Exception as e:
        logger.error("milvus_seed_failed", error=str(e))


# --- Sample PostgreSQL User ---
async def seed_postgresql() -> None:
    """테스트 사용자 생성."""
    logger = get_logger("seed")
    try:
        from src.data.repositories.session import SessionRepository
        from passlib.hash import bcrypt

        repo = SessionRepository()

        # Check if user already exists
        existing = await repo.get_user_by_email("admin@example.com")
        if existing:
            logger.info("test_user_already_exists")
            return

        password_hash = bcrypt.hash("admin1234")
        user = await repo.create_user(
            email="admin@example.com",
            password_hash=password_hash,
            name="Admin User",
        )
        logger.info("postgresql_user_seeded", user_id=user.id)
    except Exception as e:
        logger.error("postgresql_seed_failed", error=str(e))


async def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger("seed")
    logger.info("seeding_starting")

    await seed_postgresql()
    await seed_influxdb()
    await seed_neo4j()
    await seed_milvus()

    logger.info("seeding_complete")


if __name__ == "__main__":
    asyncio.run(main())
