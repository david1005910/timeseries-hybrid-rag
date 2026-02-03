"""WebSocket handlers for streaming responses."""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.api.dependencies import get_coordinator
from src.data.models.schemas import QueryRequest
from src.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


@router.websocket("/ws/query")
async def websocket_query(websocket: WebSocket) -> None:
    """WebSocket 스트리밍 질의 처리.

    Client sends: {"query": "...", "session_id": "...", "options": {...}}
    Server sends progress updates and final answer as JSON chunks.
    """
    await websocket.accept()
    logger.info("websocket_connected")

    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)

            # Send progress updates
            await websocket.send_json({"type": "progress", "step": "planning", "message": "실행 계획 수립 중..."})

            request = QueryRequest(**request_data)
            coordinator = get_coordinator()

            # Process query
            await websocket.send_json({"type": "progress", "step": "retrieving", "message": "관련 정보 검색 중..."})

            response = await coordinator.process_query(request)

            await websocket.send_json({"type": "progress", "step": "reasoning", "message": "추론 및 검증 중..."})

            # Send final result
            await websocket.send_json({
                "type": "result",
                "data": response.model_dump(),
            })

    except WebSocketDisconnect:
        logger.info("websocket_disconnected")
    except Exception as e:
        logger.error("websocket_error", error=str(e))
        error_msg = str(e)
        if "AuthenticationError" in error_msg or "No LLM provider configured" in error_msg:
            error_msg = "LLM API 인증 실패: .env 파일에 유효한 ANTHROPIC_API_KEY 또는 OPENAI_API_KEY를 설정하세요."
        await websocket.send_json({"type": "error", "message": error_msg})
