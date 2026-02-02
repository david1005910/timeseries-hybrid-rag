"""Session Manager: conversation session lifecycle management."""
from __future__ import annotations

from typing import Any

from src.data.repositories.session import SessionRepository
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SessionManager:
    """세션 관리자: 대화 세션 생명주기 관리."""

    def __init__(self, session_repo: SessionRepository) -> None:
        self._repo = session_repo

    async def get_or_create_session(self, user_id: str, session_id: str | None = None) -> str:
        """기존 세션 조회 또는 새 세션 생성."""
        if session_id:
            existing = await self._repo.get_session(session_id)
            if existing:
                return existing.id
            logger.warning("session_not_found", session_id=session_id)

        session = await self._repo.create_session(user_id=user_id)
        logger.info("session_created", session_id=session.id, user_id=user_id)
        return session.id

    async def get_user_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """사용자의 세션 목록 조회."""
        sessions = await self._repo.get_sessions(user_id)
        return [
            {
                "id": s.id,
                "title": s.title,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "updated_at": s.updated_at.isoformat() if s.updated_at else None,
            }
            for s in sessions
        ]

    async def get_session_messages(self, session_id: str) -> list[dict[str, Any]]:
        """세션 메시지 조회."""
        messages = await self._repo.get_messages(session_id)
        return [
            {
                "role": m.role,
                "content": m.content,
                "sources": m.sources,
                "confidence": m.confidence,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ]

    async def delete_session(self, session_id: str) -> bool:
        """세션 삭제."""
        deleted = await self._repo.delete_session(session_id)
        if deleted:
            logger.info("session_deleted", session_id=session_id)
        return deleted
