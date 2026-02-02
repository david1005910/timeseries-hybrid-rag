"""Rate limiting middleware."""
from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse, Response

from src.utils.logging import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting 미들웨어.

    Per-IP 기반 요청 제한 (초당 50 요청).
    """

    def __init__(self, app, max_requests: int = 50, window_seconds: int = 1) -> None:  # type: ignore[no-untyped-def]
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._request_counts: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Clean old entries
        self._request_counts[client_ip] = [
            t for t in self._request_counts[client_ip] if now - t < self.window_seconds
        ]

        if len(self._request_counts[client_ip]) >= self.max_requests:
            logger.warning("rate_limit_exceeded", client_ip=client_ip)
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Try again later."},
            )

        self._request_counts[client_ip].append(now)
        return await call_next(request)
