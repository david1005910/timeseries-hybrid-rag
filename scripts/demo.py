"""Standalone demo: runs the full RAG pipeline without external services.

All databases and LLM calls are simulated in-memory so no Docker, no API keys,
and no network access is required.

Usage:
    poetry run python scripts/demo.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.models.schemas import QueryRequest, QueryResponse
from src.orchestration.coordinator import AgentCoordinator
from src.utils.logging import setup_logging, get_logger


# =============================================================================
# In-Memory Sample Data
# =============================================================================

SAMPLE_TIMESERIES = [
    {"time": "2025-01-31T14:25:00Z", "field": "cpu_usage", "value": 32.5, "tags": {"host": "server-01"}},
    {"time": "2025-01-31T14:28:00Z", "field": "cpu_usage", "value": 35.2, "tags": {"host": "server-01"}},
    {"time": "2025-01-31T14:30:00Z", "field": "cpu_usage", "value": 72.8, "tags": {"host": "server-01"}},
    {"time": "2025-01-31T14:31:00Z", "field": "cpu_usage", "value": 88.3, "tags": {"host": "server-01"}},
    {"time": "2025-01-31T14:32:00Z", "field": "cpu_usage", "value": 95.1, "tags": {"host": "server-01"}},
    {"time": "2025-01-31T14:33:00Z", "field": "cpu_usage", "value": 91.7, "tags": {"host": "server-01"}},
    {"time": "2025-01-31T14:35:00Z", "field": "cpu_usage", "value": 78.4, "tags": {"host": "server-01"}},
    {"time": "2025-01-31T14:40:00Z", "field": "cpu_usage", "value": 45.0, "tags": {"host": "server-01"}},
    {"time": "2025-01-31T14:30:00Z", "field": "memory_usage", "value": 70.2, "tags": {"host": "server-01"}},
    {"time": "2025-01-31T14:32:00Z", "field": "memory_usage", "value": 88.5, "tags": {"host": "server-01"}},
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
        "content": "데이터베이스 연결 풀 고갈은 트래픽이 평소 대비 300% 증가한 시점에서 발생했습니다. max_connections=50 설정이 증가된 트래픽을 감당하지 못했습니다.",
        "source": "postmortem-2025-01-31",
        "source_type": "vector",
        "relevance_score": 0.82,
        "metadata": {"type": "postmortem"},
    },
    {
        "id": "doc-004",
        "content": "권장 조치: 1) 데이터베이스 연결 풀 크기를 150으로 증가 2) 배치 작업 스케줄링 개선 3) Auto-scaling 임계값 조정",
        "source": "postmortem-2025-01-31",
        "source_type": "vector",
        "relevance_score": 0.75,
        "metadata": {"type": "recommendation"},
    },
]

SAMPLE_GRAPH_PATHS = [
    {
        "nodes": [
            {"id": "ev1", "name": "batch-job-7842", "type": "event"},
            {"id": "ev3", "name": "memory-pressure", "type": "event"},
            {"id": "ev2", "name": "cpu-spike", "type": "event"},
            {"id": "ev4", "name": "service-degradation", "type": "event"},
        ],
        "rels": [
            {"type": "CAUSES", "confidence": 0.90},
            {"type": "CAUSES", "confidence": 0.88},
            {"type": "CAUSES", "confidence": 0.92},
        ],
        "hops": 3,
    },
    {
        "nodes": [
            {"id": "ev6", "name": "traffic-spike-300pct", "type": "event"},
            {"id": "ev5", "name": "connection-pool-exhaustion", "type": "event"},
            {"id": "ev1", "name": "batch-job-7842", "type": "event"},
        ],
        "rels": [
            {"type": "CAUSES", "confidence": 0.85},
            {"type": "CAUSES", "confidence": 0.75},
        ],
        "hops": 2,
    },
]


# =============================================================================
# Mock Factories
# =============================================================================

def make_mock_llm() -> MagicMock:
    """LLM 클라이언트 mock: 프롬프트 내용 기반 응답 라우팅."""
    from src.llm.client import LLMClient, LLMResponse

    client = MagicMock(spec=LLMClient)

    # Extractor response (reused for all documents)
    _extractor_resp = json.dumps({
        "entities": [
            {"name": "batch-job-7842", "type": "event", "properties": {"severity": "critical"}},
            {"name": "server-01", "type": "entity", "properties": {"status": "degraded"}},
            {"name": "cpu_usage", "type": "metric", "properties": {"peak": "95%"}},
        ],
        "relationships": [
            {"source": "batch-job-7842", "target": "cpu_usage", "type": "causes", "confidence": 0.92},
        ],
    })

    _planner_resp = json.dumps({
        "intent": "시계열 이상 탐지 + 근본 원인 분석",
        "complexity": "high",
        "language": "ko",
        "data_sources": ["timeseries", "vector", "graph"],
        "estimated_hops": 3,
        "steps": [
            {"agent": "retriever", "params": {"sources": ["timeseries", "vector", "graph"]}},
            {"agent": "extractor", "params": {}},
            {"agent": "reasoner", "params": {"include_graph": True}},
            {"agent": "validator", "params": {}},
        ],
        "clarification_needed": False,
    })

    _reasoner_resp = json.dumps({
        "answer": (
            "CPU 급증의 근본 원인은 14:30에 시작된 대규모 배치 작업(batch-7842)입니다.\n\n"
            "## 인과관계 분석\n"
            "1. **트래픽 급증** (14:25): 평소 대비 300% 트래픽 증가가 발생\n"
            "2. **연결 풀 고갈** (14:29): DB 연결 풀(max=50)이 포화 상태에 도달\n"
            "3. **배치 작업 시작** (14:30): batch-7842가 데이터 집계 시작, 평소 3배 데이터 처리\n"
            "4. **메모리 압박** (14:31): 임시 데이터 적재로 메모리 70%→88% 급증\n"
            "5. **CPU 스파이크** (14:32): GC 압력 + 배치 연산으로 CPU 95% 도달\n"
            "6. **서비스 장애** (14:33): API 응답 지연 5초 초과, 503 에러 발생\n\n"
            "## 권장 조치\n"
            "- DB 연결 풀 크기 50 → 150 증가\n"
            "- 배치 작업을 피크 시간 외로 스케줄링 변경\n"
            "- Auto-scaling 임계값 CPU 80% → 70%로 하향 조정\n"
            "- 메모리 할당 최적화 (GC 튜닝)"
        ),
        "confidence": 0.89,
        "reasoning_steps": [
            "시계열 데이터에서 14:30-14:35 구간 CPU 95% 스파이크 확인",
            "지식 그래프에서 batch-7842 → memory-pressure → cpu-spike 인과 체인 발견",
            "벡터 검색 문서에서 배치 작업이 평소 3배 데이터 처리 확인",
            "인과관계 체인 종합: traffic-spike → connection-pool → batch-job → memory → cpu → degradation",
        ],
        "causal_chain": [
            {"source": "traffic-spike-300pct", "relation": "CAUSES", "confidence": 0.85, "target": "connection-pool-exhaustion"},
            {"source": "connection-pool-exhaustion", "relation": "CAUSES", "confidence": 0.75, "target": "batch-job-7842"},
            {"source": "batch-job-7842", "relation": "CAUSES", "confidence": 0.90, "target": "memory-pressure"},
            {"source": "memory-pressure", "relation": "CAUSES", "confidence": 0.88, "target": "cpu-spike"},
            {"source": "cpu-spike", "relation": "CAUSES", "confidence": 0.92, "target": "service-degradation"},
        ],
        "uncertainties": [],
    })

    _verifier_resp = json.dumps({
        "retrieve": {"decision": "No", "reason": "충분한 증거가 확보됨"},
        "is_relevant": [
            {"source_idx": 1, "decision": "Relevant", "reason": "배치 작업과 CPU 급증 직접 연관"},
            {"source_idx": 2, "decision": "Relevant", "reason": "메모리 증가와 GC 부하 설명"},
            {"source_idx": 3, "decision": "Relevant", "reason": "연결 풀 고갈 원인 제시"},
        ],
        "is_supported": {
            "level": "Fully",
            "supported_parts": ["배치 작업 원인", "메모리 압박", "인과관계 체인"],
            "unsupported_parts": [],
        },
        "is_useful": {"score": 5, "reason": "근본 원인, 인과관계, 권장 조치 모두 포함"},
        "final_verdict": "pass",
        "adjusted_answer": None,
    })

    async def _generate(prompt: str, **kwargs: Any) -> LLMResponse:
        # Route by prompt content
        if "실행 계획을 수립하세요" in prompt:
            content = _planner_resp
        elif "엔티티와 관계를 추출하세요" in prompt:
            content = _extractor_resp
        elif "Chain-of-Thought" in prompt or "단계별" in prompt and "추론" in prompt:
            content = _reasoner_resp
        elif "Self-RAG" in prompt or "Reflection Token" in prompt:
            content = _verifier_resp
        else:
            content = json.dumps({"result": "ok"})

        return LLMResponse(
            content=content,
            provider="demo-mock",
            model="demo-model",
            input_tokens=500,
            output_tokens=400,
            elapsed_ms=150.0,
        )

    client.generate = _generate
    return client


def make_mock_embedding() -> MagicMock:
    from src.llm.embeddings import EmbeddingService
    svc = MagicMock(spec=EmbeddingService)
    svc.embed = AsyncMock(return_value=[0.1] * 1536)
    svc.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
    return svc


def make_mock_timeseries() -> MagicMock:
    from src.data.repositories.timeseries import TimeseriesRepository
    repo = MagicMock(spec=TimeseriesRepository)
    repo.query_metrics = AsyncMock(return_value=SAMPLE_TIMESERIES)
    return repo


def make_mock_vector() -> MagicMock:
    from src.data.repositories.vector import VectorRepository
    repo = MagicMock(spec=VectorRepository)
    repo.search = AsyncMock(return_value=[
        {
            "id": doc["id"],
            "score": 1.0 - i * 0.05,
            "content": doc["content"],
            "source": doc["source"],
            "source_type": doc["source_type"],
            "metadata_json": json.dumps(doc["metadata"]),
        }
        for i, doc in enumerate(SAMPLE_DOCUMENTS)
    ])
    return repo


def make_mock_graph() -> MagicMock:
    from src.data.repositories.graph import GraphRepository
    repo = MagicMock(spec=GraphRepository)
    repo.traverse = AsyncMock(return_value=SAMPLE_GRAPH_PATHS)
    repo.find_causal_chain = AsyncMock(return_value=SAMPLE_GRAPH_PATHS)
    repo.get_node = AsyncMock(return_value={"id": "ev1", "name": "batch-job-7842", "type": "event"})

    mock_session = AsyncMock()

    async def _aiter():
        for rec in [{"id": "ev1", "name": "batch-job-7842", "labels": ["event"]}]:
            yield rec

    mock_session.run = AsyncMock(return_value=_aiter())

    class _Ctx:
        async def __aenter__(self):
            return mock_session
        async def __aexit__(self, *args):
            return None

    repo._get_session = AsyncMock(return_value=_Ctx())
    return repo


def make_mock_session() -> MagicMock:
    from src.data.repositories.session import SessionRepository
    repo = MagicMock(spec=SessionRepository)
    repo.get_messages = AsyncMock(return_value=[])
    repo.add_message = AsyncMock()
    return repo


# =============================================================================
# Pretty Print
# =============================================================================

def print_header(text: str) -> None:
    width = 70
    print(f"\n{'=' * width}")
    print(f"  {text}")
    print(f"{'=' * width}")


def print_section(title: str) -> None:
    print(f"\n--- {title} ---")


def print_response(resp: QueryResponse) -> None:
    print_section("Answer")
    print(resp.answer)

    print_section(f"Confidence: {resp.confidence:.2f}")

    if resp.reasoning_chain:
        print_section("Reasoning Chain")
        for step in resp.reasoning_chain:
            dur = f"{step.duration_ms:.0f}ms" if step.duration_ms else ""
            conf = f" (confidence: {step.confidence:.2f})" if step.confidence else ""
            print(f"  Step {step.step}. [{step.action}] {step.result} {dur}{conf}")

    if resp.sources:
        print_section(f"Sources ({len(resp.sources)})")
        for src in resp.sources:
            print(f"  [{src.source_type}] {src.source_id}: {src.content[:80]}...")

    if resp.graph_path:
        print_section("Causal Chain (Graph)")
        for path in resp.graph_path:
            if isinstance(path, dict):
                src = path.get("source", "?")
                rel = path.get("relation", "?")
                conf = path.get("confidence", 0)
                tgt = path.get("target", "?")
                print(f"  {src} --[{rel} {conf}]--> {tgt}")
            else:
                print(f"  {path}")

    if resp.warnings:
        print_section("Warnings")
        for w in resp.warnings:
            print(f"  ! {w}")

    print_section(f"Processing Time: {resp.processing_time_ms:.0f}ms")


# =============================================================================
# Main
# =============================================================================

async def run_demo() -> None:
    setup_logging("INFO")
    logger = get_logger("demo")

    print_header("Hybrid RAG System - Demo")
    print("Docker/외부 서비스 없이 인메모리 시뮬레이션으로 실행합니다.")
    print("모든 LLM 응답은 미리 작성된 시나리오 데이터입니다.")

    # Build coordinator with mocks
    coordinator = AgentCoordinator(
        llm_client=make_mock_llm(),
        embedding_service=make_mock_embedding(),
        timeseries_repo=make_mock_timeseries(),
        vector_repo=make_mock_vector(),
        graph_repo=make_mock_graph(),
        session_repo=make_mock_session(),
    )

    # ---- Query 1 ----
    print_header("Query 1: CPU 급증 근본 원인 분석")
    query1 = QueryRequest(
        query="어제 서버 CPU 사용률이 급증한 원인을 분석해줘",
        options={"max_hops": 5, "include_reasoning": True},
    )
    print(f"Q: {query1.query}\n")

    t0 = time.time()
    resp1 = await coordinator.process_query(query1, user_id="demo-user")
    print_response(resp1)

    # ---- Query 2 ----
    # Reset mock LLM for second query
    coordinator._llm = make_mock_llm_query2()
    coordinator._planner._llm = coordinator._llm
    coordinator._extractor._llm = coordinator._llm
    coordinator._reasoner._llm = coordinator._llm
    coordinator._validator._llm = coordinator._llm
    coordinator._verifier._llm = coordinator._llm

    print_header("Query 2: 메모리 사용량 추세 분석")
    query2 = QueryRequest(
        query="최근 메모리 사용량 추세와 이상 징후를 분석해줘",
    )
    print(f"Q: {query2.query}\n")

    resp2 = await coordinator.process_query(query2, user_id="demo-user")
    print_response(resp2)

    # ---- Summary ----
    print_header("Demo Summary")
    print(f"  Queries executed:  2")
    print(f"  Total pipeline steps: {len(resp1.reasoning_chain) + len(resp2.reasoning_chain)}")
    print(f"  Data sources used: timeseries, vector, graph (in-memory)")
    print(f"  Self-RAG verification: passed")
    print(f"  All results simulated - no external services needed")


def make_mock_llm_query2() -> MagicMock:
    """Second query LLM mock: prompt-content-based routing."""
    from src.llm.client import LLMClient, LLMResponse

    client = MagicMock(spec=LLMClient)

    _planner_resp = json.dumps({
        "intent": "메모리 사용량 추세 분석",
        "complexity": "medium",
        "language": "ko",
        "data_sources": ["timeseries", "vector"],
        "estimated_hops": 2,
        "steps": [
            {"agent": "retriever", "params": {"sources": ["timeseries", "vector"]}},
            {"agent": "reasoner", "params": {}},
            {"agent": "validator", "params": {}},
        ],
        "clarification_needed": False,
    })

    _reasoner_resp = json.dumps({
        "answer": (
            "메모리 사용량 분석 결과, 1월 31일 14:30-14:35 구간에서 이상 징후가 감지되었습니다.\n\n"
            "## 추세 분석\n"
            "- **정상 범위**: 55-65% (평균 60%)\n"
            "- **이상 구간**: 14:30~14:35에 70% → 88%로 급증 (+18%p)\n"
            "- **복구 시점**: 14:40 이후 정상 범위 복귀\n\n"
            "## 원인\n"
            "배치 작업(batch-7842)이 대량 임시 데이터를 메모리에 적재하면서 "
            "GC 압력이 증가했습니다. 이는 CPU 급증과 동시에 발생했으며, "
            "두 메트릭 간 상관계수가 0.78로 높은 상관관계를 보입니다."
        ),
        "confidence": 0.84,
        "reasoning_steps": [
            "시계열 데이터에서 메모리 사용량 추세 분석",
            "정상 범위(55-65%) 대비 이상 구간(70-88%) 식별",
            "벡터 검색으로 관련 장애 보고서 확인",
        ],
        "causal_chain": [
            {"source": "batch-job-7842", "relation": "CAUSES", "confidence": 0.90, "target": "memory-pressure"},
            {"source": "memory-pressure", "relation": "CORRELATES", "confidence": 0.78, "target": "cpu-spike"},
        ],
        "uncertainties": [],
    })

    _verifier_resp = json.dumps({
        "retrieve": {"decision": "No", "reason": "충분한 증거"},
        "is_relevant": [
            {"source_idx": 1, "decision": "Relevant", "reason": "메모리 관련"},
        ],
        "is_supported": {
            "level": "Fully",
            "supported_parts": ["메모리 추세", "배치 작업 원인"],
            "unsupported_parts": [],
        },
        "is_useful": {"score": 4, "reason": "추세와 원인 분석 포함"},
        "final_verdict": "pass",
        "adjusted_answer": None,
    })

    async def _generate(prompt: str, **kwargs: Any) -> LLMResponse:
        if "실행 계획을 수립하세요" in prompt:
            content = _planner_resp
        elif "엔티티와 관계를 추출하세요" in prompt:
            content = json.dumps({"entities": [], "relationships": []})
        elif "Chain-of-Thought" in prompt or "단계별" in prompt and "추론" in prompt:
            content = _reasoner_resp
        elif "Self-RAG" in prompt or "Reflection Token" in prompt:
            content = _verifier_resp
        else:
            content = json.dumps({"result": "ok"})

        return LLMResponse(
            content=content,
            provider="demo-mock",
            model="demo-model",
            input_tokens=400,
            output_tokens=350,
            elapsed_ms=120.0,
        )

    client.generate = _generate
    return client


if __name__ == "__main__":
    asyncio.run(run_demo())
