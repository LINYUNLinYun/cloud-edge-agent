"""Chat service — business orchestration for the chat endpoint.

Flow: Request → Session → Memory Retrieval → Orchestrator → Memory Store → Response

Privacy-aware storage routing:
  - S1 (Safe) → Qdrant cloud (long-term memory)
  - S2/S3 (Sensitive/Confidential) → SQLite local (never leaves device)

Session cache architecture:
  - In-memory cache for fast access
  - Automatic persistence based on privacy level
"""

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.core.logger.logger import get_logger
from app.domain.memory.memory import MemoryEntry, MemoryStore
from app.infrastructure.cache.session_cache import SessionCacheManager
from app.infrastructure.database.session_repository import InMemorySessionStore
from app.infrastructure.rag.pipeline import RAGPipeline
from app.services.agent_orchestrator import (
    CollaborativeOrchestrator,
    OrchestratorResult,
)

logger = get_logger(__name__)

# Maximum number of recent messages to inject as context
_MAX_CONTEXT_MESSAGES = 10


@dataclass
class ChatResponse:
    """Response from the chat service."""

    answer: str
    session_id: str
    mode: str
    privacy_level: str
    complexity: int
    latency_ms: float


class ChatService:
    """Orchestrates a chat request through the full pipeline.

    Uses SessionCacheManager for:
    - Fast in-memory session access
    - Privacy-aware persistence (S1→Qdrant, S2/S3→SQLite)
    - Unified search across all memory sources
    """

    def __init__(
        self,
        orchestrator: CollaborativeOrchestrator,
        session_store: InMemorySessionStore,
        short_term_memory: MemoryStore,
        cloud_memory: MemoryStore | None = None,
        local_memory: MemoryStore | None = None,
        rag_pipeline: RAGPipeline | None = None,
    ) -> None:
        self._orchestrator = orchestrator
        self._sessions = session_store
        self._short_term = short_term_memory
        self._rag = rag_pipeline

        # Session cache with privacy-aware persistence
        self._cache = SessionCacheManager(
            cloud_memory=cloud_memory,
            local_memory=local_memory,
        )

    async def chat(
        self, query: str, session_id: str | None = None
    ) -> ChatResponse:
        """Process a chat message end-to-end.

        1. Ensure session exists
        2. Retrieve context from cache + persistent storage
        3. Enrich query with context
        4. Run orchestrator (privacy → complexity → route → execute)
        5. Store in cache + persist based on privacy level
        6. Return structured response
        """
        # Ensure session
        if session_id is None:
            import uuid

            session_id = uuid.uuid4().hex[:12]
        if self._sessions.get(session_id) is None:
            self._sessions.create(session_id)

        # Retrieve context from cache + persistent storage
        context = await self._cache.get_context(
            session_id=session_id,
            query=query,
            max_entries=_MAX_CONTEXT_MESSAGES,
        )

        # Also get RAG context if available
        if self._rag is not None:
            try:
                rag_results = await self._rag.retrieve(query, top_k=3)
                for result in rag_results:
                    context.append(
                        MemoryEntry(
                            content=result.document.content,
                            metadata={
                                **result.document.metadata,
                                "source": "rag",
                                "score": result.score,
                            },
                            session_id=session_id,
                        )
                    )
            except Exception as exc:
                logger.warning("rag_retrieval_failed", error=str(exc))

        enriched_query = self._enrich_query(query, context)

        # Run orchestrator
        result: OrchestratorResult = await self._orchestrator.process(
            query=enriched_query, session_id=session_id
        )

        # Store in session store (for API compatibility)
        self._sessions.add_message(session_id, "user", query)
        self._sessions.add_message(session_id, "assistant", result.answer)

        # Store in cache + persist based on privacy level
        privacy_level = result.routing.privacy_level.value

        user_entry = MemoryEntry(
            content=query,
            session_id=session_id,
            metadata={"role": "user", "privacy_level": privacy_level},
        )
        assistant_entry = MemoryEntry(
            content=result.answer,
            session_id=session_id,
            metadata={
                "role": "assistant",
                "mode": result.mode.value,
                "privacy_level": privacy_level,
            },
        )
        conversation_entry = MemoryEntry(
            content=f"User: {query}\nAssistant: {result.answer}",
            session_id=session_id,
            metadata={"mode": result.mode.value, "privacy_level": privacy_level},
        )

        # Add to cache (handles persistence automatically)
        await self._cache.add(session_id, user_entry, privacy_level)
        await self._cache.add(session_id, assistant_entry, privacy_level)

        # Also persist the full conversation pair
        await self._cache.persist_conversation(session_id, conversation_entry, privacy_level)

        logger.info(
            "chat_complete",
            session_id=session_id,
            mode=result.mode.value,
            privacy_level=privacy_level,
            latency_ms=result.latency_ms,
            context_messages=len(context),
            cache_stats=self._cache.get_stats(),
        )

        return ChatResponse(
            answer=result.answer,
            session_id=session_id,
            mode=result.mode.value,
            privacy_level=privacy_level,
            complexity=result.routing.complexity.value,
            latency_ms=result.latency_ms,
        )

    async def chat_stream(
        self, query: str, session_id: str | None = None
    ) -> AsyncIterator[str]:
        """Process a chat message and stream the response.

        This runs the full orchestrator pipeline (privacy detection, routing,
        agent execution) and then streams the final answer token by token
        for a better user experience.

        Yields:
            JSON strings with either token data or metadata.
        """
        # Ensure session
        if session_id is None:
            import uuid

            session_id = uuid.uuid4().hex[:12]
        if self._sessions.get(session_id) is None:
            self._sessions.create(session_id)

        # Retrieve context
        context = await self._cache.get_context(
            session_id=session_id,
            query=query,
            max_entries=_MAX_CONTEXT_MESSAGES,
        )
        enriched_query = self._enrich_query(query, context)

        # Run orchestrator
        result: OrchestratorResult = await self._orchestrator.process(
            query=enriched_query, session_id=session_id
        )

        # Store in session store
        self._sessions.add_message(session_id, "user", query)
        self._sessions.add_message(session_id, "assistant", result.answer)

        # Store in cache + persist
        privacy_level = result.routing.privacy_level.value
        user_entry = MemoryEntry(
            content=query,
            session_id=session_id,
            metadata={"role": "user", "privacy_level": privacy_level},
        )
        assistant_entry = MemoryEntry(
            content=result.answer,
            session_id=session_id,
            metadata={"role": "assistant", "mode": result.mode.value, "privacy_level": privacy_level},
        )
        await self._cache.add(session_id, user_entry, privacy_level)
        await self._cache.add(session_id, assistant_entry, privacy_level)

        # Stream metadata first
        metadata = {
            "type": "metadata",
            "session_id": session_id,
            "mode": result.mode.value,
            "privacy_level": privacy_level,
            "complexity": result.routing.complexity.value,
            "latency_ms": result.latency_ms,
        }
        yield json.dumps(metadata, ensure_ascii=False)

        # Stream the answer token by token (simulated for demo)
        answer = result.answer
        chunk_size = 3  # characters per chunk
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i : i + chunk_size]
            yield json.dumps({"type": "token", "delta": chunk}, ensure_ascii=False)
            await asyncio.sleep(0.02)  # Small delay for visual effect

        # Signal completion
        yield json.dumps({"type": "done"})

        logger.info(
            "chat_stream_complete",
            session_id=session_id,
            mode=result.mode.value,
            latency_ms=result.latency_ms,
        )

    @staticmethod
    def _enrich_query(query: str, context: list[MemoryEntry]) -> str:
        """Prepend relevant memory context to the user query.

        The enriched query gives the LLM access to conversation history
        without requiring it to be in the message window.
        """
        if not context:
            return query

        context_lines = []
        for entry in context:
            role = entry.metadata.get("role", "unknown")
            # Truncate long entries to avoid overwhelming the LLM
            content = entry.content[:500] if len(entry.content) > 500 else entry.content
            context_lines.append(f"[{role}] {content}")

        context_block = "\n".join(context_lines)

        # Limit total context length to avoid token limits
        max_context_len = 2000
        if len(context_block) > max_context_len:
            context_block = context_block[:max_context_len] + "\n... (context truncated)"

        logger.info(
            "enrich_query",
            context_entries=len(context),
            context_len=len(context_block),
        )

        return (
            f"Relevant conversation history:\n{context_block}\n\n"
            f"Current question: {query}"
        )
