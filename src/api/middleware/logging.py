"""Request logging middleware."""
from __future__ import annotations

import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.utils.logging import correlation_id_var, get_logger

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """요청 로깅 미들웨어 with Correlation ID."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())[:8]
        correlation_id_var.set(correlation_id)

        start_time = time.time()

        logger.info(
            "request_start",
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        response = await call_next(request)

        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            "request_end",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            elapsed_ms=round(elapsed_ms, 2),
        )

        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Processing-Time-Ms"] = str(round(elapsed_ms, 2))

        return response
