"""Integration tests for the query endpoint."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_coordinator, get_current_user_id
from src.api.routes.query import router
from src.data.models.schemas import QueryResponse
from src.orchestration.coordinator import AgentCoordinator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_coordinator() -> MagicMock:
    """Create a mock AgentCoordinator that returns a valid QueryResponse."""
    coord = MagicMock(spec=AgentCoordinator)
    coord.process_query = AsyncMock(
        return_value=QueryResponse(
            id="test-query-id",
            answer="배치 작업이 CPU 급증의 원인입니다.",
            confidence=0.85,
            reasoning_chain=[],
            sources=[],
            graph_path=[],
            processing_time_ms=123.45,
            session_id=None,
            warnings=[],
        )
    )
    return coord


@pytest.fixture
def app() -> FastAPI:
    """Minimal FastAPI app with query router and dependency overrides."""
    test_app = FastAPI()
    test_app.include_router(router)

    mock_coord = _make_mock_coordinator()
    test_app.dependency_overrides[get_coordinator] = lambda: mock_coord
    test_app.dependency_overrides[get_current_user_id] = lambda: "test-user"

    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture
def app_with_failing_coordinator() -> FastAPI:
    """App where the coordinator raises an exception."""
    test_app = FastAPI()
    test_app.include_router(router)

    coord = MagicMock(spec=AgentCoordinator)
    coord.process_query = AsyncMock(side_effect=RuntimeError("LLM provider unavailable"))
    test_app.dependency_overrides[get_coordinator] = lambda: coord
    test_app.dependency_overrides[get_current_user_id] = lambda: "test-user"

    return test_app


@pytest.fixture
def failing_client(app_with_failing_coordinator: FastAPI) -> TestClient:
    return TestClient(app_with_failing_coordinator)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQueryEndpoint:
    def test_query_success(self, client: TestClient) -> None:
        """POST /api/v1/query with a valid body should return 200 with an answer."""
        response = client.post(
            "/api/v1/query",
            json={"query": "어제 CPU 급증 원인을 분석해줘"},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "배치 작업이 CPU 급증의 원인입니다."
        assert body["confidence"] == 0.85
        assert "processing_time_ms" in body

    def test_query_response_structure(self, client: TestClient) -> None:
        """Response must contain all fields defined in QueryResponse schema."""
        response = client.post(
            "/api/v1/query",
            json={"query": "테스트 질의"},
        )

        body = response.json()
        required_keys = {"id", "answer", "confidence", "reasoning_chain", "sources",
                         "graph_path", "processing_time_ms", "session_id", "warnings"}
        assert required_keys.issubset(set(body.keys()))

    def test_query_empty_body_returns_422(self, client: TestClient) -> None:
        """POST /api/v1/query with missing query field should return 422."""
        response = client.post("/api/v1/query", json={})

        assert response.status_code == 422

    def test_query_empty_string_returns_422(self, client: TestClient) -> None:
        """POST /api/v1/query with empty string query should return 422 (min_length=1)."""
        response = client.post(
            "/api/v1/query",
            json={"query": ""},
        )

        assert response.status_code == 422

    def test_query_coordinator_failure_returns_500(self, failing_client: TestClient) -> None:
        """When the coordinator raises, the endpoint should return HTTP 500."""
        response = failing_client.post(
            "/api/v1/query",
            json={"query": "이 질의는 실패할 것입니다"},
        )

        assert response.status_code == 500
        body = response.json()
        assert "Query processing failed" in body["detail"]

    def test_query_with_options(self, client: TestClient) -> None:
        """POST /api/v1/query should accept optional session_id and options."""
        response = client.post(
            "/api/v1/query",
            json={
                "query": "CPU 급증 원인은?",
                "session_id": "sess-123",
                "options": {
                    "max_hops": 3,
                    "include_reasoning": True,
                    "stream": False,
                    "language": "ko",
                },
            },
        )

        assert response.status_code == 200


class TestFeedbackEndpoint:
    def test_feedback_returns_success(self, client: TestClient) -> None:
        """POST /api/v1/query/{query_id}/feedback should return status."""
        response = client.post(
            "/api/v1/query/test-query-id/feedback",
            json={
                "query_id": "test-query-id",
                "rating": 4,
                "comment": "유용한 답변이었습니다",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "feedback_received"
        assert body["query_id"] == "test-query-id"
