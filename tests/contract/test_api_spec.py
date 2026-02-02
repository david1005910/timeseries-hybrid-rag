"""Contract tests verifying API endpoints match the specification.

These tests validate that:
1. All specified endpoints exist and are reachable
2. Response schemas match Pydantic models
3. Request validation works as expected
4. HTTP methods and status codes follow the spec
"""
from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import (
    get_coordinator,
    get_graph_repo,
    get_session_manager,
    get_timeseries_repo,
)
from src.data.models.schemas import QueryResponse
from src.main import create_app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_coordinator() -> MagicMock:
    mock = MagicMock()
    mock.process_query = AsyncMock(
        return_value=QueryResponse(
            answer="테스트 답변",
            confidence=0.85,
            processing_time_ms=500.0,
        )
    )
    return mock


def _make_mock_session_mgr() -> MagicMock:
    mgr = MagicMock()
    mgr.get_user_sessions = AsyncMock(return_value=[])
    mgr.get_or_create_session = AsyncMock(return_value="sess-new")
    mgr.get_session_messages = AsyncMock(return_value=[])
    mgr.delete_session = AsyncMock(return_value=True)
    return mgr


def _make_mock_graph_repo() -> MagicMock:
    repo = MagicMock()
    mock_session = AsyncMock()

    async def _empty_aiter():
        return
        yield  # noqa: make async generator

    mock_session.run = AsyncMock(return_value=_empty_aiter())

    class _FakeCtx:
        async def __aenter__(self):
            return mock_session
        async def __aexit__(self, *args):
            return None

    repo._get_session = AsyncMock(return_value=_FakeCtx())
    repo.traverse = AsyncMock(return_value=[])
    return repo


def _make_mock_ts_repo() -> MagicMock:
    repo = MagicMock()
    repo.query_metrics = AsyncMock(return_value=[])
    return repo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app() -> FastAPI:
    """Full application with dependency overrides for all external services."""
    application = create_app()
    application.dependency_overrides[get_coordinator] = _make_mock_coordinator
    application.dependency_overrides[get_session_manager] = _make_mock_session_mgr
    application.dependency_overrides[get_graph_repo] = _make_mock_graph_repo
    application.dependency_overrides[get_timeseries_repo] = _make_mock_ts_repo
    return application


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health API Contract: GET /api/v1/health
# ---------------------------------------------------------------------------

class TestHealthContract:
    """Health endpoint must return status, version, and services dict."""

    def test_endpoint_exists(self, client: TestClient) -> None:
        with patch("src.data.repositories.session.get_engine", return_value=MagicMock()):
            resp = client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_response_schema(self, client: TestClient) -> None:
        with patch("src.data.repositories.session.get_engine", return_value=MagicMock()):
            body = client.get("/api/v1/health").json()
        assert isinstance(body["status"], str)
        assert isinstance(body["version"], str)
        assert isinstance(body["services"], dict)


# ---------------------------------------------------------------------------
# Query API Contract: POST /api/v1/query
# ---------------------------------------------------------------------------

class TestQueryContract:
    """POST /api/v1/query must accept QueryRequest and return QueryResponse."""

    def test_valid_query_returns_200(self, client: TestClient) -> None:
        resp = client.post("/api/v1/query", json={"query": "CPU 급증 원인은?"})
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client: TestClient) -> None:
        body = client.post("/api/v1/query", json={"query": "테스트"}).json()
        required = {"id", "answer", "confidence", "processing_time_ms"}
        assert required.issubset(set(body.keys()))

    def test_empty_query_rejected(self, client: TestClient) -> None:
        resp = client.post("/api/v1/query", json={"query": ""})
        assert resp.status_code == 422

    def test_missing_query_rejected(self, client: TestClient) -> None:
        resp = client.post("/api/v1/query", json={})
        assert resp.status_code == 422

    def test_query_too_long_rejected(self, client: TestClient) -> None:
        resp = client.post("/api/v1/query", json={"query": "a" * 2001})
        assert resp.status_code == 422

    def test_optional_session_id_accepted(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/query",
            json={"query": "테스트", "session_id": "sess-123"},
        )
        assert resp.status_code == 200

    def test_optional_options_accepted(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/query",
            json={
                "query": "테스트",
                "options": {"max_hops": 3, "include_reasoning": False},
            },
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Feedback API Contract: POST /api/v1/query/{id}/feedback
# ---------------------------------------------------------------------------

class TestFeedbackContract:
    """POST /api/v1/query/{id}/feedback must accept FeedbackRequest."""

    def test_valid_feedback_returns_200(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/query/q-123/feedback",
            json={"query_id": "q-123", "rating": 4, "comment": "유용했습니다"},
        )
        assert resp.status_code == 200

    def test_feedback_response_structure(self, client: TestClient) -> None:
        body = client.post(
            "/api/v1/query/q-123/feedback",
            json={"query_id": "q-123", "rating": 3},
        ).json()
        assert body["status"] == "feedback_received"
        assert body["query_id"] == "q-123"

    def test_rating_validation_min(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/query/q-123/feedback",
            json={"query_id": "q-123", "rating": 0},
        )
        assert resp.status_code == 422

    def test_rating_validation_max(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/query/q-123/feedback",
            json={"query_id": "q-123", "rating": 6},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Session API Contract: /api/v1/sessions
# ---------------------------------------------------------------------------

class TestSessionContract:
    """Session endpoints must handle CRUD operations."""

    def test_list_sessions_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/sessions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_create_session_returns_200(self, client: TestClient) -> None:
        resp = client.post("/api/v1/sessions")
        assert resp.status_code == 200
        assert "session_id" in resp.json()

    def test_get_session_messages_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/sessions/sess-1")
        assert resp.status_code == 200

    def test_delete_session_returns_200(self, client: TestClient) -> None:
        resp = client.delete("/api/v1/sessions/sess-1")
        assert resp.status_code == 200

    def test_delete_nonexistent_session_returns_404(self, app: FastAPI, client: TestClient) -> None:
        mgr = _make_mock_session_mgr()
        mgr.delete_session = AsyncMock(return_value=False)
        app.dependency_overrides[get_session_manager] = lambda: mgr
        resp = client.delete("/api/v1/sessions/ghost")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Graph API Contract: /api/v1/graph
# ---------------------------------------------------------------------------

class TestGraphContract:
    """Graph endpoints must support entity listing and traversal."""

    def test_list_entities_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/graph/entities")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_list_entities_with_type_filter(self, client: TestClient) -> None:
        resp = client.get("/api/v1/graph/entities?node_type=metric")
        assert resp.status_code == 200

    def test_list_entities_limit_validation(self, client: TestClient) -> None:
        resp = client.get("/api/v1/graph/entities?limit=201")
        assert resp.status_code == 422

    def test_traverse_returns_200(self, client: TestClient) -> None:
        resp = client.get("/api/v1/graph/entities/node-1/traverse")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_traverse_max_hops_validation(self, client: TestClient) -> None:
        resp = client.get("/api/v1/graph/entities/node-1/traverse?max_hops=6")
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Metrics API Contract: POST /api/v1/metrics/query
# ---------------------------------------------------------------------------

class TestMetricsContract:
    """Metrics query endpoint must accept MetricsQuery and return MetricsResponse."""

    def test_valid_metrics_query_returns_200(self, client: TestClient) -> None:
        resp = client.post(
            "/api/v1/metrics/query",
            json={"measurement": "system_metrics"},
        )
        assert resp.status_code == 200

    def test_metrics_response_structure(self, client: TestClient) -> None:
        body = client.post(
            "/api/v1/metrics/query",
            json={"measurement": "system_metrics"},
        ).json()
        assert body["measurement"] == "system_metrics"
        assert isinstance(body["records"], list)
        assert "count" in body
        assert "query_time_ms" in body

    def test_metrics_missing_measurement_rejected(self, client: TestClient) -> None:
        resp = client.post("/api/v1/metrics/query", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# OpenAPI Schema Contract
# ---------------------------------------------------------------------------

class TestOpenAPISchema:
    """The auto-generated OpenAPI schema must contain all documented endpoints."""

    EXPECTED_PATHS = [
        "/api/v1/health",
        "/api/v1/query",
        "/api/v1/query/{query_id}/feedback",
        "/api/v1/sessions",
        "/api/v1/sessions/{session_id}",
        "/api/v1/graph/entities",
        "/api/v1/graph/entities/{node_id}/traverse",
        "/api/v1/metrics/query",
    ]

    def test_openapi_schema_accessible(self, client: TestClient) -> None:
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        assert "openapi" in resp.json()

    def test_all_spec_endpoints_registered(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        paths = set(schema["paths"].keys())
        for expected in self.EXPECTED_PATHS:
            assert expected in paths, f"Missing endpoint: {expected}"

    def test_correct_http_methods(self, client: TestClient) -> None:
        schema = client.get("/openapi.json").json()
        paths = schema["paths"]
        assert "get" in paths["/api/v1/health"]
        assert "post" in paths["/api/v1/query"]
        assert "post" in paths["/api/v1/query/{query_id}/feedback"]
        assert "get" in paths["/api/v1/sessions"]
        assert "post" in paths["/api/v1/sessions"]
        assert "get" in paths["/api/v1/graph/entities"]
        assert "post" in paths["/api/v1/metrics/query"]
