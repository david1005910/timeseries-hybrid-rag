"""FastAPI dependency injection."""
from __future__ import annotations

from functools import lru_cache

from fastapi import Header

from src.data.repositories.graph import GraphRepository
from src.data.repositories.session import SessionRepository
from src.data.repositories.timeseries import TimeseriesRepository
from src.data.repositories.vector import VectorRepository
from src.llm.client import LLMClient
from src.llm.embeddings import EmbeddingService
from src.orchestration.coordinator import AgentCoordinator
from src.orchestration.session_manager import SessionManager


@lru_cache
def get_llm_client() -> LLMClient:
    return LLMClient()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache
def get_timeseries_repo() -> TimeseriesRepository:
    return TimeseriesRepository()


@lru_cache
def get_vector_repo() -> VectorRepository:
    return VectorRepository()


@lru_cache
def get_graph_repo() -> GraphRepository:
    return GraphRepository()


@lru_cache
def get_session_repo() -> SessionRepository:
    return SessionRepository()


@lru_cache
def get_session_manager() -> SessionManager:
    return SessionManager(get_session_repo())


@lru_cache
def get_coordinator() -> AgentCoordinator:
    return AgentCoordinator(
        llm_client=get_llm_client(),
        embedding_service=get_embedding_service(),
        timeseries_repo=get_timeseries_repo(),
        vector_repo=get_vector_repo(),
        graph_repo=get_graph_repo(),
        session_repo=get_session_repo(),
    )


async def get_current_user_id(
    authorization: str | None = Header(None),
) -> str | None:
    """현재 사용자 ID 추출 (JWT에서). 인증이 없으면 None."""
    if not authorization:
        return None
    # TODO: JWT token verification
    # For now, return a placeholder
    return "anonymous"
