"""Health check API routes."""
from __future__ import annotations

from fastapi import APIRouter

from src.data.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/api/v1/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """서비스 상태 확인."""
    services: dict[str, str] = {}

    # Check PostgreSQL
    try:
        from src.data.repositories.session import get_engine
        engine = get_engine()
        services["postgresql"] = "ok"
    except Exception:
        services["postgresql"] = "unavailable"

    # Other services will be checked when Docker is running
    services["influxdb"] = "not_checked"
    services["neo4j"] = "not_checked"
    services["milvus"] = "not_checked"
    services["redis"] = "not_checked"

    return HealthResponse(services=services)
