"""Query API routes."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_coordinator, get_current_user_id
from src.data.models.schemas import FeedbackRequest, QueryRequest, QueryResponse
from src.orchestration.coordinator import AgentCoordinator

router = APIRouter(prefix="/api/v1", tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def process_query(
    request: QueryRequest,
    coordinator: AgentCoordinator = Depends(get_coordinator),
    user_id: str | None = Depends(get_current_user_id),
) -> QueryResponse:
    """자연어 질의 처리.

    전체 RAG 파이프라인을 실행하여 답변을 생성합니다:
    Plan → Retrieve → Reason → Validate
    """
    try:
        response = await coordinator.process_query(request, user_id=user_id)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")


@router.post("/query/{query_id}/feedback")
async def submit_feedback(
    query_id: str,
    feedback: FeedbackRequest,
    user_id: str = Depends(get_current_user_id),
) -> dict[str, str]:
    """질의 결과에 대한 피드백 제출."""
    # TODO: Implement feedback storage
    return {"status": "feedback_received", "query_id": query_id}
