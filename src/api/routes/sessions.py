"""Session API routes."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_current_user_id, get_session_manager
from src.orchestration.session_manager import SessionManager

router = APIRouter(prefix="/api/v1/sessions", tags=["sessions"])


@router.get("")
async def list_sessions(
    user_id: str = Depends(get_current_user_id),
    session_mgr: SessionManager = Depends(get_session_manager),
) -> list[dict[str, Any]]:
    """사용자의 세션 목록 조회."""
    return await session_mgr.get_user_sessions(user_id)


@router.post("")
async def create_session(
    user_id: str = Depends(get_current_user_id),
    session_mgr: SessionManager = Depends(get_session_manager),
) -> dict[str, str]:
    """새 세션 생성."""
    session_id = await session_mgr.get_or_create_session(user_id)
    return {"session_id": session_id}


@router.get("/{session_id}")
async def get_session_messages(
    session_id: str,
    session_mgr: SessionManager = Depends(get_session_manager),
) -> list[dict[str, Any]]:
    """세션 메시지 조회."""
    return await session_mgr.get_session_messages(session_id)


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    session_mgr: SessionManager = Depends(get_session_manager),
) -> dict[str, str]:
    """세션 삭제."""
    deleted = await session_mgr.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted"}
