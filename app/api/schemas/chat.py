"""API schemas for chat endpoints."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Incoming chat request."""

    query: str = Field(..., min_length=1, description="User message")
    session_id: str | None = Field(default=None, description="Session ID for context")


class ChatResponseSchema(BaseModel):
    """Outgoing chat response."""

    answer: str
    session_id: str
    mode: str = Field(description="Collaborate mode used")
    privacy_level: str = Field(description="Detected privacy level (S1/S2/S3)")
    complexity: int = Field(description="Detected complexity level (1-5)")
    latency_ms: float
    budget_remaining: float = Field(description="Remaining privacy budget ε")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"


class PrivacyBudgetResponse(BaseModel):
    """Privacy budget status for a session."""

    session_id: str
    remaining_epsilon: float
    exhausted: bool


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str = ""
