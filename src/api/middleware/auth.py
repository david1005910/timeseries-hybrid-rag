"""Authentication middleware."""
from __future__ import annotations

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.utils.logging import get_logger

logger = get_logger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """인증 미들웨어 (OAuth 2.0 + RBAC)."""

    # Public endpoints that don't require authentication
    PUBLIC_PATHS = {"/api/v1/health", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        # Skip auth for public endpoints
        if path in self.PUBLIC_PATHS or path.startswith("/static"):
            return await call_next(request)

        # TODO: Implement JWT verification
        # For now, pass through all requests
        return await call_next(request)
