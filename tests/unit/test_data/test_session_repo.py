"""Tests for SessionRepository (PostgreSQL / SQLAlchemy async).

Uses an in-memory SQLite database via aiosqlite so that tests run without
a real PostgreSQL instance. The module-level ``_engine`` and ``_session_factory``
globals in the session module are patched to use the in-memory engine.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest
from sqlalchemy import event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.data.models.entities import Base
from src.data.repositories.session import SessionRepository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def engine():
    """Create an in-memory SQLite async engine for testing."""
    test_engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    # Enable foreign key support in SQLite (off by default)
    @event.listens_for(test_engine.sync_engine, "connect")
    def _set_sqlite_fk(dbapi_conn, _connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield test_engine

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await test_engine.dispose()


@pytest.fixture
async def session_factory(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


@pytest.fixture
async def repo(session_factory) -> SessionRepository:
    """Patch the module-level factory so SessionRepository uses the test DB."""
    with patch(
        "src.data.repositories.session.get_session_factory",
        return_value=session_factory,
    ):
        repository = SessionRepository()
        yield repository


# ---------------------------------------------------------------------------
# User Tests
# ---------------------------------------------------------------------------

class TestUserOperations:
    async def test_create_user(self, repo: SessionRepository) -> None:
        """create_user persists a user and returns it with a generated id."""
        user = await repo.create_user(
            email="alice@example.com",
            password_hash="hashed_pw",
            name="Alice",
        )

        assert user.id is not None
        assert user.email == "alice@example.com"
        assert user.name == "Alice"
        assert user.password_hash == "hashed_pw"

    async def test_get_user_by_email_found(self, repo: SessionRepository) -> None:
        """get_user_by_email returns the matching user."""
        await repo.create_user(email="bob@example.com", password_hash="h", name="Bob")

        found = await repo.get_user_by_email("bob@example.com")

        assert found is not None
        assert found.name == "Bob"

    async def test_get_user_by_email_not_found(self, repo: SessionRepository) -> None:
        """get_user_by_email returns None for unknown addresses."""
        assert await repo.get_user_by_email("nobody@example.com") is None


# ---------------------------------------------------------------------------
# Chat Session Tests
# ---------------------------------------------------------------------------

class TestChatSessionOperations:
    async def _create_test_user(self, repo: SessionRepository) -> str:
        user = await repo.create_user(
            email="sess-user@example.com", password_hash="h", name="SessUser"
        )
        return user.id

    async def test_create_session(self, repo: SessionRepository) -> None:
        """create_session returns a ChatSession linked to the user."""
        user_id = await self._create_test_user(repo)

        chat_session = await repo.create_session(user_id=user_id, title="Test Chat")

        assert chat_session.id is not None
        assert chat_session.user_id == user_id
        assert chat_session.title == "Test Chat"

    async def test_get_sessions(self, repo: SessionRepository) -> None:
        """get_sessions returns all sessions for a given user."""
        user_id = await self._create_test_user(repo)
        await repo.create_session(user_id=user_id, title="Session A")
        await repo.create_session(user_id=user_id, title="Session B")

        sessions = await repo.get_sessions(user_id)

        assert len(sessions) == 2
        titles = {s.title for s in sessions}
        assert titles == {"Session A", "Session B"}

    async def test_get_session_by_id(self, repo: SessionRepository) -> None:
        """get_session returns the specific session or None."""
        user_id = await self._create_test_user(repo)
        created = await repo.create_session(user_id=user_id, title="Specific")

        found = await repo.get_session(created.id)

        assert found is not None
        assert found.title == "Specific"

    async def test_get_session_not_found(self, repo: SessionRepository) -> None:
        """get_session returns None for a non-existent id."""
        assert await repo.get_session("nonexistent-id") is None

    async def test_delete_session(self, repo: SessionRepository) -> None:
        """delete_session removes the session and returns True."""
        user_id = await self._create_test_user(repo)
        created = await repo.create_session(user_id=user_id, title="ToDelete")

        deleted = await repo.delete_session(created.id)

        assert deleted is True
        assert await repo.get_session(created.id) is None

    async def test_delete_session_not_found(self, repo: SessionRepository) -> None:
        """delete_session returns False when no session matches."""
        assert await repo.delete_session("ghost-id") is False


# ---------------------------------------------------------------------------
# Chat Message Tests
# ---------------------------------------------------------------------------

class TestChatMessageOperations:
    async def _create_session(self, repo: SessionRepository) -> str:
        user = await repo.create_user(
            email="msg-user@example.com", password_hash="h", name="MsgUser"
        )
        chat_session = await repo.create_session(user_id=user.id, title="Msg Session")
        return chat_session.id

    async def test_add_message(self, repo: SessionRepository) -> None:
        """add_message persists a message and returns it with an id."""
        session_id = await self._create_session(repo)

        msg = await repo.add_message(
            session_id=session_id,
            role="user",
            content="What caused the CPU spike?",
        )

        assert msg.id is not None
        assert msg.role == "user"
        assert msg.content == "What caused the CPU spike?"
        assert msg.session_id == session_id

    async def test_add_message_with_metadata(self, repo: SessionRepository) -> None:
        """add_message stores optional sources, confidence, and processing time."""
        session_id = await self._create_session(repo)

        msg = await repo.add_message(
            session_id=session_id,
            role="assistant",
            content="The spike was caused by a batch job.",
            sources=[{"id": "doc-1", "score": 0.92}],
            confidence=0.88,
            processing_time_ms=1234.5,
        )

        assert msg.confidence == 0.88
        assert msg.processing_time_ms == 1234.5

    async def test_get_messages_ordered(self, repo: SessionRepository) -> None:
        """get_messages returns messages for a session in chronological order."""
        session_id = await self._create_session(repo)
        await repo.add_message(session_id=session_id, role="user", content="First")
        await repo.add_message(session_id=session_id, role="assistant", content="Second")
        await repo.add_message(session_id=session_id, role="user", content="Third")

        messages = await repo.get_messages(session_id)

        assert len(messages) == 3
        assert messages[0].content == "First"
        assert messages[1].content == "Second"
        assert messages[2].content == "Third"

    async def test_get_messages_empty(self, repo: SessionRepository) -> None:
        """get_messages returns an empty list for a session with no messages."""
        session_id = await self._create_session(repo)

        messages = await repo.get_messages(session_id)

        assert messages == []


# ---------------------------------------------------------------------------
# Feedback Tests
# ---------------------------------------------------------------------------

class TestFeedbackOperations:
    async def test_add_feedback(self, repo: SessionRepository) -> None:
        """add_feedback persists a QueryFeedback row and returns it."""
        user = await repo.create_user(
            email="fb-user@example.com", password_hash="h", name="FBUser"
        )

        feedback = await repo.add_feedback(
            query_id="query-1",
            user_id=user.id,
            rating=4,
            comment="Great answer",
            correction=None,
        )

        assert feedback.id is not None
        assert feedback.query_id == "query-1"
        assert feedback.rating == 4
        assert feedback.comment == "Great answer"

    async def test_add_feedback_with_correction(self, repo: SessionRepository) -> None:
        """add_feedback stores an optional correction field."""
        user = await repo.create_user(
            email="fb-user2@example.com", password_hash="h", name="FBUser2"
        )

        feedback = await repo.add_feedback(
            query_id="query-2",
            user_id=user.id,
            rating=2,
            comment="Not quite right",
            correction="The actual cause was a memory leak, not a batch job.",
        )

        assert feedback.correction is not None
        assert "memory leak" in feedback.correction
