"""Integration tests for the health-check endpoint."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.health import router


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app() -> FastAPI:
    """Minimal FastAPI app with only the health router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client: TestClient) -> None:
        """GET /api/v1/health should return HTTP 200."""
        with patch("src.data.repositories.session.get_engine") as mock_engine:
            mock_engine.return_value = MagicMock()
            response = client.get("/api/v1/health")

        assert response.status_code == 200

    def test_health_response_structure(self, client: TestClient) -> None:
        """Response body must include status, version, and services dict."""
        with patch("src.data.repositories.session.get_engine") as mock_engine:
            mock_engine.return_value = MagicMock()
            response = client.get("/api/v1/health")

        body = response.json()
        assert "status" in body
        assert "version" in body
        assert "services" in body
        assert isinstance(body["services"], dict)

    def test_health_status_ok(self, client: TestClient) -> None:
        """Default status field should be 'ok'."""
        with patch("src.data.repositories.session.get_engine") as mock_engine:
            mock_engine.return_value = MagicMock()
            response = client.get("/api/v1/health")

        body = response.json()
        assert body["status"] == "ok"
        assert body["version"] == "0.1.0"

    def test_health_postgres_ok_when_engine_succeeds(self, client: TestClient) -> None:
        """When get_engine succeeds, postgresql should be 'ok'."""
        with patch("src.data.repositories.session.get_engine") as mock_engine:
            mock_engine.return_value = MagicMock()
            response = client.get("/api/v1/health")

        services = response.json()["services"]
        assert services["postgresql"] == "ok"

    def test_health_postgres_unavailable_when_engine_fails(self, client: TestClient) -> None:
        """When get_engine raises, postgresql should be 'unavailable'."""
        with patch(
            "src.data.repositories.session.get_engine",
            side_effect=Exception("db down"),
        ):
            response = client.get("/api/v1/health")

        services = response.json()["services"]
        assert services["postgresql"] == "unavailable"
        # Other services are still reported
        assert services["influxdb"] == "not_checked"
        assert services["neo4j"] == "not_checked"
        assert services["milvus"] == "not_checked"
        assert services["redis"] == "not_checked"

    def test_health_services_contain_all_expected_keys(self, client: TestClient) -> None:
        """The services dict should contain all five backend service entries."""
        with patch("src.data.repositories.session.get_engine") as mock_engine:
            mock_engine.return_value = MagicMock()
            response = client.get("/api/v1/health")

        services = response.json()["services"]
        expected_keys = {"postgresql", "influxdb", "neo4j", "milvus", "redis"}
        assert expected_keys.issubset(set(services.keys()))
