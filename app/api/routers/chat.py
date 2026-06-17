"""Chat API router."""

from fastapi import APIRouter, HTTPException, Request

from app.api.schemas.chat import (
    ChatRequest,
    ChatResponseSchema,
    ErrorResponse,
    PrivacyBudgetResponse,
)
from app.core.exceptions.exceptions import BaseAppException

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post(
    "",
    response_model=ChatResponseSchema,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def chat(request: Request, body: ChatRequest) -> ChatResponseSchema:
    """Send a message and get a response from the cloud-edge agent.

    The system automatically:
    - Detects privacy level (S1/S2/S3)
    - Analyzes task complexity (L1-L5)
    - Routes to edge or cloud with appropriate collaborate mode
    """
    chat_service = request.app.state.components.chat_service
    try:
        result = await chat_service.chat(
            query=body.query, session_id=body.session_id
        )
        return ChatResponseSchema(
            answer=result.answer,
            session_id=result.session_id,
            mode=result.mode,
            privacy_level=result.privacy_level,
            complexity=result.complexity,
            latency_ms=result.latency_ms,
            budget_remaining=result.budget_remaining,
        )
    except BaseAppException as exc:
        raise HTTPException(status_code=400, detail=exc.message) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get(
    "/budget/{session_id}",
    response_model=PrivacyBudgetResponse,
)
async def get_budget(request: Request, session_id: str) -> PrivacyBudgetResponse:
    """Get the remaining privacy budget for a session."""
    budget_tracker = request.app.state.components.budget_tracker
    remaining = budget_tracker.get_remaining(session_id)
    return PrivacyBudgetResponse(
        session_id=session_id,
        remaining_epsilon=remaining,
        exhausted=budget_tracker.is_exhausted(session_id),
    )
