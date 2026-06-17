"""Session repository — in-memory storage for conversation sessions.

Replace with PostgreSQL/SQLite for production persistence.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.core.logger.logger import get_logger
from app.domain.memory.memory import MemoryEntry, MemoryStore, MemoryType

logger = get_logger(__name__)


@dataclass
class Session:
    """A conversation session."""

    session_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    messages: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class InMemorySessionStore:
    """In-memory session storage (for MVP)."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, session_id: str) -> Session:
        """Create a new session."""
        session = Session(session_id=session_id)
        self._sessions[session_id] = session
        logger.info("session_created", session_id=session_id)
        return session

    def get(self, session_id: str) -> Session | None:
        """Retrieve a session by ID."""
        return self._sessions.get(session_id)

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session history."""
        session = self._sessions.get(session_id)
        if session:
            session.messages.append({"role": role, "content": content})

    def list_sessions(self) -> list[Session]:
        """List all sessions."""
        return list(self._sessions.values())


class InMemoryShortTermStore(MemoryStore):
    """In-memory short-term memory (conversation context window)."""

    memory_type = MemoryType.SHORT_TERM

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    async def add(self, entry: MemoryEntry) -> str:
        entry.entry_id = str(len(self._entries))
        self._entries.append(entry)
        return entry.entry_id

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        # Simple keyword matching for short-term (no embedding needed)
        query_lower = query.lower()
        scored = [
            (e, 1.0 if query_lower in e.content.lower() else 0.0)
            for e in self._entries
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [e for e, _ in scored[:top_k]]

    async def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        return self._entries[-limit:]

    async def clear(self, session_id: str | None = None) -> None:
        if session_id:
            self._entries = [e for e in self._entries if e.session_id != session_id]
        else:
            self._entries.clear()
