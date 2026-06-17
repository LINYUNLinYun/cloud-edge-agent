"""Chat service — business orchestration for the chat endpoint.

Flow: Request → Session → Orchestrator → Memory → Response
"""

from dataclasses import dataclass

from app.core.logger.logger import get_logger
from app.domain.memory.memory import MemoryEntry, MemoryStore
from app.infrastructure.database.session_repository import InMemorySessionStore
from app.services.agent_orchestrator import (
    CollaborativeOrchestrator,
    OrchestratorResult,
)

logger = get_logger(__name__)


@dataclass
class ChatResponse:
    """Response from the chat service."""

    answer: str
    session_id: str
    mode: str
    privacy_level: str
    complexity: int
    latency_ms: float
    budget_remaining: float


class ChatService:
    """Orchestrates a chat request through the full pipeline."""

    def __init__(
        self,
        orchestrator: CollaborativeOrchestrator,
        session_store: InMemorySessionStore,
        short_term_memory: MemoryStore,
        budget_tracker,  # PrivacyBudgetTracker
    ) -> None:
        self._orchestrator = orchestrator
        self._sessions = session_store
        self._memory = short_term_memory
        self._budget = budget_tracker

    async def chat(
        self, query: str, session_id: str | None = None
    ) -> ChatResponse:
        """Process a chat message end-to-end.

        1. Ensure session exists
        2. Store user message in memory
        3. Run orchestrator (privacy → complexity → route → execute)
        4. Store assistant response in memory
        5. Return structured response
        """
        # Ensure session
        if session_id is None:
            import uuid

            session_id = uuid.uuid4().hex[:12]
        if self._sessions.get(session_id) is None:
            self._sessions.create(session_id)

        # Store user message
        self._sessions.add_message(session_id, "user", query)
        await self._memory.add(
            MemoryEntry(content=query, session_id=session_id, metadata={"role": "user"})
        )

        # Run orchestrator
        result: OrchestratorResult = await self._orchestrator.process(
            query=query, session_id=session_id
        )

        # Store assistant response
        self._sessions.add_message(session_id, "assistant", result.answer)
        await self._memory.add(
            MemoryEntry(
                content=result.answer,
                session_id=session_id,
                metadata={"role": "assistant", "mode": result.mode.value},
            )
        )

        logger.info(
            "chat_complete",
            session_id=session_id,
            mode=result.mode.value,
            latency_ms=result.latency_ms,
        )

        return ChatResponse(
            answer=result.answer,
            session_id=session_id,
            mode=result.mode.value,
            privacy_level=result.routing.privacy_level.value,
            complexity=result.routing.complexity.value,
            latency_ms=result.latency_ms,
            budget_remaining=self._budget.get_remaining(session_id),
        )
