"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.routes import graph, health, metrics, query, sessions
from src.api.websocket.handlers import router as ws_router
from src.config.settings import get_settings
from src.data.repositories.session import init_db
from src.utils.logging import setup_logging, get_logger


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifecycle management."""
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = get_logger("main")
    logger.info("application_starting", env=settings.app_env)

    # Initialize database tables
    try:
        await init_db()
        logger.info("database_initialized")
    except Exception as e:
        logger.warning("database_init_failed", error=str(e))

    yield

    logger.info("application_shutting_down")


def create_app() -> FastAPI:
    """FastAPI 앱 생성 및 설정."""
    settings = get_settings()

    app = FastAPI(
        title="Hybrid RAG System",
        description="GraphRAG + Self-RAG + Multi-Agent 하이브리드 추론 강화 RAG 시스템",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Middleware (order matters: last added = first executed)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RateLimitMiddleware, max_requests=50, window_seconds=1)

    # REST Routes
    app.include_router(health.router)
    app.include_router(query.router)
    app.include_router(sessions.router)
    app.include_router(metrics.router)
    app.include_router(graph.router)

    # WebSocket
    app.include_router(ws_router)

    return app


app = create_app()
