"""Live test: runs the RAG pipeline with a real LLM API key.

Data repositories are mocked in-memory, but LLM calls go to the actual
Anthropic API so we can verify end-to-end generation, fallback, and
Self-RAG verification with real model responses.

Usage:
    poetry run python scripts/live_test.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import get_settings
from src.data.models.schemas import QueryRequest, QueryResponse
from src.llm.client import LLMClient, LLMResponse
from src.orchestration.coordinator import AgentCoordinator
from src.utils.logging import setup_logging, get_logger


# =============================================================================
# In-Memory Sample Data (same as demo.py)
# =============================================================================

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
]

SAMPLE_TIMESERIES = [
    {"time": "2025-01-31T14:25:00Z", "field": "cpu_usage", "value": 32.5, "tags": {"host": "server-01"}},
    {"time": "2025-01-31T14:30:00Z", "field": "cpu_usage", "value": 72.8, "tags": {"host": "server-01"}},
    {"time": "2025-01-31T14:32:00Z", "field": "cpu_usage", "value": 95.1, "tags": {"host": "server-01"}},
    {"time": "2025-01-31T14:35:00Z", "field": "cpu_usage", "value": 78.4, "tags": {"host": "server-01"}},
    {"time": "2025-01-31T14:40:00Z", "field": "cpu_usage", "value": 45.0, "tags": {"host": "server-01"}},
]


# =============================================================================
# Mock factories (data layer only - LLM is real)
# =============================================================================

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
    repo.traverse = AsyncMock(return_value=[])
    repo.find_causal_chain = AsyncMock(return_value=[])
    repo.get_node = AsyncMock(return_value=None)
    repo.create_node = AsyncMock()
    repo.create_relationship = AsyncMock()

    mock_session = AsyncMock()

    async def _aiter():
        return
        yield  # noqa: make async generator

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


def print_response(resp: QueryResponse) -> None:
    print(f"\n--- Answer ---")
    print(resp.answer)

    print(f"\n--- Confidence: {resp.confidence:.2f} ---")

    if resp.reasoning_chain:
        print(f"\n--- Reasoning Chain ---")
        for step in resp.reasoning_chain:
            dur = f"{step.duration_ms:.0f}ms" if step.duration_ms else ""
            conf = f" (confidence: {step.confidence:.2f})" if step.confidence else ""
            print(f"  Step {step.step}. [{step.action}] {step.result} {dur}{conf}")

    if resp.sources:
        print(f"\n--- Sources ({len(resp.sources)}) ---")
        for src in resp.sources:
            print(f"  [{src.source_type}] {src.source_id}: {src.content[:80]}...")

    if resp.graph_path:
        print(f"\n--- Causal Chain (Graph) ---")
        for path in resp.graph_path:
            if isinstance(path, dict):
                s = path.get("source", "?")
                r = path.get("relation", "?")
                c = path.get("confidence", 0)
                t = path.get("target", "?")
                print(f"  {s} --[{r} {c}]--> {t}")
            else:
                print(f"  {path}")

    if resp.warnings:
        print(f"\n--- Warnings ---")
        for w in resp.warnings:
            print(f"  ! {w}")

    print(f"\n--- Processing Time: {resp.processing_time_ms:.0f}ms ---")


# =============================================================================
# Live Tests
# =============================================================================

async def test_1_direct_llm(llm: LLMClient) -> None:
    """Test 1: Direct LLM call to verify API key works."""
    print_header("Test 1: Direct LLM API Call")
    print("Anthropic Claude에 직접 요청을 보냅니다...\n")

    t0 = time.time()
    response = await llm.generate(
        prompt="서버 CPU 사용률이 95%까지 급증했을 때 가능한 원인 3가지를 간단히 알려줘.",
        system_prompt="당신은 인프라 모니터링 전문가입니다. 간결하게 답변하세요.",
        temperature=0.3,
        max_tokens=500,
    )
    elapsed = (time.time() - t0) * 1000

    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Tokens: {response.input_tokens} in / {response.output_tokens} out")
    print(f"Elapsed: {elapsed:.0f}ms")
    print(f"Cost: ${response.estimated_cost_usd:.6f}")
    print(f"\n--- Response ---")
    print(response.content)
    print("\n[PASS] LLM API 연결 성공")


async def test_2_full_pipeline(llm: LLMClient) -> None:
    """Test 2: Full RAG pipeline with real LLM."""
    print_header("Test 2: Full RAG Pipeline (Real LLM)")
    print("실제 LLM으로 전체 파이프라인을 실행합니다...\n")

    coordinator = AgentCoordinator(
        llm_client=llm,
        embedding_service=make_mock_embedding(),
        timeseries_repo=make_mock_timeseries(),
        vector_repo=make_mock_vector(),
        graph_repo=make_mock_graph(),
        session_repo=make_mock_session(),
    )

    query = QueryRequest(query="어제 서버 CPU 사용률이 급증한 원인을 분석해줘")
    print(f"Q: {query.query}\n")

    resp = await coordinator.process_query(query, user_id="live-test-user")
    print_response(resp)

    passed = resp.answer and len(resp.answer) > 10 and resp.confidence > 0
    status = "PASS" if passed else "FAIL"
    print(f"\n[{status}] Full pipeline {'성공' if passed else '실패'}")
    return passed


async def test_3_streaming(llm: LLMClient) -> None:
    """Test 3: Streaming response."""
    print_header("Test 3: LLM Streaming")
    print("스트리밍 응답을 테스트합니다...\n")

    chunks: list[str] = []
    t0 = time.time()
    async for text in llm.generate_stream(
        prompt="Redis 캐시를 사용해야 하는 이유 2가지를 한 문장씩 알려줘.",
        system_prompt="간결하게 답변하세요.",
        max_tokens=200,
    ):
        chunks.append(text)
        print(text, end="", flush=True)

    elapsed = (time.time() - t0) * 1000
    print(f"\n\nChunks received: {len(chunks)}")
    print(f"Total length: {sum(len(c) for c in chunks)} chars")
    print(f"Elapsed: {elapsed:.0f}ms")
    print(f"\n[PASS] Streaming 성공")


# =============================================================================
# Main
# =============================================================================

async def run_live_test() -> None:
    setup_logging("WARNING")  # suppress structlog noise
    logger = get_logger("live_test")

    print_header("Hybrid RAG System - Live Test")
    print("실제 Anthropic API를 사용한 라이브 테스트입니다.")
    print("데이터 레이어는 인메모리 mock, LLM만 실제 API 호출합니다.\n")

    # Verify API key
    settings = get_settings()
    if not settings.anthropic_api_key or settings.anthropic_api_key.startswith("your-"):
        print("[ERROR] ANTHROPIC_API_KEY가 설정되지 않았습니다.")
        print("  .env 파일에 유효한 키를 설정하세요.")
        sys.exit(1)

    print(f"Anthropic API Key: ...{settings.anthropic_api_key[-8:]}")
    print(f"OpenAI API Key: {'설정됨' if settings.openai_api_key else '없음 (fallback 비활성)'}")

    llm = LLMClient()

    results: list[tuple[str, bool]] = []

    # Test 1: Direct LLM
    try:
        await test_1_direct_llm(llm)
        results.append(("Direct LLM Call", True))
    except Exception as e:
        print(f"\n[FAIL] Direct LLM: {e}")
        results.append(("Direct LLM Call", False))

    # Test 2: Full Pipeline
    try:
        passed = await test_2_full_pipeline(llm)
        results.append(("Full RAG Pipeline", passed))
    except Exception as e:
        print(f"\n[FAIL] Full Pipeline: {e}")
        results.append(("Full RAG Pipeline", False))

    # Test 3: Streaming
    try:
        await test_3_streaming(llm)
        results.append(("LLM Streaming", True))
    except Exception as e:
        print(f"\n[FAIL] Streaming: {e}")
        results.append(("LLM Streaming", False))

    # Summary
    print_header("Live Test Summary")
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        icon = "O" if passed else "X"
        print(f"  [{icon}] {name}: {status}")
        if not passed:
            all_passed = False

    print(f"\n  Result: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"  Provider: Anthropic Claude (live)")
    print(f"  Data Layer: In-memory mock")


if __name__ == "__main__":
    asyncio.run(run_live_test())
