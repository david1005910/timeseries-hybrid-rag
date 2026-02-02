"""PostgreSQL Repository for session and user data."""
from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config.settings import get_settings
from src.data.models.entities import Base, ChatMessage, ChatSession, QueryFeedback, User
from src.utils.logging import get_logger

logger = get_logger(__name__)

_engine = None
_session_factory = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(settings.database_url, echo=False, pool_size=10)
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(get_engine(), expire_on_commit=False)
    return _session_factory


async def init_db() -> None:
    """데이터베이스 테이블 생성."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_initialized")


class SessionRepository:
    """세션 및 사용자 데이터를 위한 PostgreSQL Repository."""

    def __init__(self) -> None:
        self._factory = get_session_factory()

    # --- User ---
    async def create_user(self, email: str, password_hash: str, name: str) -> User:
        async with self._factory() as session:
            user = User(email=email, password_hash=password_hash, name=name)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            return user

    async def get_user_by_email(self, email: str) -> User | None:
        async with self._factory() as session:
            result = await session.execute(select(User).where(User.email == email))
            return result.scalar_one_or_none()

    # --- Chat Session ---
    async def create_session(self, user_id: str, title: str = "New Session") -> ChatSession:
        async with self._factory() as session:
            chat_session = ChatSession(user_id=user_id, title=title)
            session.add(chat_session)
            await session.commit()
            await session.refresh(chat_session)
            return chat_session

    async def get_sessions(self, user_id: str) -> list[ChatSession]:
        async with self._factory() as session:
            result = await session.execute(
                select(ChatSession)
                .where(ChatSession.user_id == user_id)
                .order_by(ChatSession.updated_at.desc())
            )
            return list(result.scalars().all())

    async def get_session(self, session_id: str) -> ChatSession | None:
        async with self._factory() as session:
            result = await session.execute(
                select(ChatSession).where(ChatSession.id == session_id)
            )
            return result.scalar_one_or_none()

    async def delete_session(self, session_id: str) -> bool:
        async with self._factory() as session:
            chat_session = await session.get(ChatSession, session_id)
            if chat_session:
                await session.delete(chat_session)
                await session.commit()
                return True
            return False

    # --- Chat Messages ---
    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
        confidence: float | None = None,
        processing_time_ms: float | None = None,
    ) -> ChatMessage:
        async with self._factory() as session:
            msg = ChatMessage(
                session_id=session_id,
                role=role,
                content=content,
                sources=sources or [],
                confidence=confidence,
                processing_time_ms=processing_time_ms,
            )
            session.add(msg)
            await session.commit()
            await session.refresh(msg)
            return msg

    async def get_messages(self, session_id: str) -> list[ChatMessage]:
        async with self._factory() as session:
            result = await session.execute(
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at.asc())
            )
            return list(result.scalars().all())

    # --- Feedback ---
    async def add_feedback(
        self, query_id: str, user_id: str, rating: int, comment: str | None = None, correction: str | None = None
    ) -> QueryFeedback:
        async with self._factory() as session:
            feedback = QueryFeedback(
                query_id=query_id, user_id=user_id, rating=rating, comment=comment, correction=correction
            )
            session.add(feedback)
            await session.commit()
            await session.refresh(feedback)
            return feedback
