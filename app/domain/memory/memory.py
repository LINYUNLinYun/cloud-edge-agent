"""Memory abstractions — short-term (conversation) and long-term (vector) memory."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"


@dataclass
class MemoryEntry:
    """A single memory record."""

    content: str
    metadata: dict = field(default_factory=dict)
    session_id: str = ""
    entry_id: str = ""


class MemoryStore(ABC):
    """Abstract memory storage.

    Implementations: InMemoryStore (short-term), VectorMemoryStore (long-term).
    """

    memory_type: MemoryType

    @abstractmethod
    async def add(self, entry: MemoryEntry) -> str:
        """Store a memory entry. Returns entry ID."""
        ...

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Retrieve relevant memories by semantic similarity."""
        ...

    @abstractmethod
    async def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        """Get the most recent memories (for short-term context window)."""
        ...

    @abstractmethod
    async def clear(self, session_id: str | None = None) -> None:
        """Clear memories, optionally scoped to a session."""
        ...
