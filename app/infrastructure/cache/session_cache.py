"""Session cache — in-memory cache with privacy-aware persistence.

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    SessionCacheManager                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  In-Memory Cache │    │  Session Store  │                │
│  │  (LRU, fast)     │    │  (full history) │                │
│  └────────┬────────┘    └────────┬────────┘                │
│           │                      │                          │
│           ▼                      ▼                          │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              Privacy Router                          │   │
│  │  S1 → Qdrant Cloud    S2/S3 → SQLite Local          │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
"""

import time
from collections import OrderedDict
from dataclasses import dataclass, field

from app.core.logger.logger import get_logger
from app.domain.memory.memory import MemoryEntry, MemoryStore

logger = get_logger(__name__)

# Default cache settings
DEFAULT_MAX_SESSIONS = 100
DEFAULT_MAX_ENTRIES_PER_SESSION = 50


@dataclass
class SessionData:
    """Cached session data."""

    session_id: str
    entries: list[MemoryEntry] = field(default_factory=list)
    last_accessed: float = field(default_factory=time.time)
    total_messages: int = 0


class SessionCacheManager:
    """Manages session caching with privacy-aware persistence.

    Features:
    - LRU cache for fast session access
    - Automatic persistence based on privacy level
    - Unified search across cache and persistent storage
    """

    def __init__(
        self,
        cloud_memory: MemoryStore | None = None,
        local_memory: MemoryStore | None = None,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        max_entries_per_session: int = DEFAULT_MAX_ENTRIES_PER_SESSION,
    ) -> None:
        """Initialize session cache manager.

        Args:
            cloud_memory: Qdrant store for S1 conversations.
            local_memory: SQLite store for S2/S3 conversations.
            max_sessions: Maximum sessions to keep in cache.
            max_entries_per_session: Maximum entries per cached session.
        """
        self._cloud_memory = cloud_memory
        self._local_memory = local_memory
        self._max_sessions = max_sessions
        self._max_entries = max_entries_per_session
        self._cache: OrderedDict[str, SessionData] = OrderedDict()

    def _get_or_create_session(self, session_id: str) -> SessionData:
        """Get session from cache or create new one."""
        if session_id in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(session_id)
            session = self._cache[session_id]
            session.last_accessed = time.time()
            return session

        # Create new session
        session = SessionData(session_id=session_id)
        self._cache[session_id] = session

        # Evict oldest if over limit
        while len(self._cache) > self._max_sessions:
            evicted_id, evicted_session = self._cache.popitem(last=False)
            logger.info("session_evicted", session_id=evicted_id)
            # Persist evicted session to storage
            self._persist_session(evicted_session)

        return session

    def _persist_session(self, session: SessionData) -> None:
        """Persist session entries to appropriate storage."""
        # This is called when sessions are evicted from cache
        # Actual persistence happens in add() based on privacy level
        logger.info(
            "session_persisted",
            session_id=session.session_id,
            entries=session.total_messages,
        )

    async def add(
        self,
        session_id: str,
        entry: MemoryEntry,
        privacy_level: str = "S1",
    ) -> None:
        """Add entry to session cache and persist based on privacy level.

        Args:
            session_id: session identifier.
            entry: memory entry to add.
            privacy_level: S1/S2/S3 - determines storage location.
        """
        # Add to cache
        session = self._get_or_create_session(session_id)
        session.entries.append(entry)
        session.total_messages += 1

        # Trim if over limit
        if len(session.entries) > self._max_entries:
            session.entries = session.entries[-self._max_entries:]

        # Persist based on privacy level
        if privacy_level == "S1":
            # Safe → cloud storage
            if self._cloud_memory is not None:
                await self._cloud_memory.add(entry)
                logger.debug("persisted_to_cloud", session_id=session_id)
        else:
            # S2/S3 → local storage
            if self._local_memory is not None:
                await self._local_memory.add(entry)
                logger.debug("persisted_to_local", session_id=session_id)

    async def get_context(
        self,
        session_id: str,
        query: str,
        max_entries: int = 10,
    ) -> list[MemoryEntry]:
        """Get relevant context from cache and persistent storage.

        Searches:
        1. In-memory cache (fast, current session)
        2. Cloud memory (S1 past sessions)
        3. Local memory (S2/S3 past sessions)

        Args:
            session_id: current session.
            query: search query.
            max_entries: maximum context entries.

        Returns:
            List of relevant MemoryEntry objects.
        """
        context: list[MemoryEntry] = []

        # 1. Current session from cache (most recent, always relevant)
        if session_id in self._cache:
            session = self._cache[session_id]
            # Add recent entries from current session
            recent = session.entries[-max_entries:]
            context.extend(recent)

        # 2. Cloud memory (S1 past conversations)
        if self._cloud_memory is not None:
            try:
                cloud_hits = await self._cloud_memory.search(
                    query, top_k=max_entries
                )
                # Filter out current session entries (already in cache)
                cloud_hits = [
                    h for h in cloud_hits if h.session_id != session_id
                ]
                context.extend(cloud_hits)
            except Exception as exc:
                logger.warning("cloud_search_failed", error=str(exc))

        # 3. Local memory (S2/S3 past conversations)
        if self._local_memory is not None:
            try:
                local_hits = await self._local_memory.search(
                    query, top_k=max_entries
                )
                # Filter out current session entries
                local_hits = [
                    h for h in local_hits if h.session_id != session_id
                ]
                context.extend(local_hits)
            except Exception as exc:
                logger.warning("local_search_failed", error=str(exc))

        logger.info(
            "context_retrieved",
            session_id=session_id,
            total=len(context),
            from_cache=sum(1 for c in context if c.session_id == session_id),
            from_cloud=sum(
                1 for c in context
                if c.session_id != session_id
                and c.metadata.get("privacy_level") == "S1"
            ),
            from_local=sum(
                1 for c in context
                if c.session_id != session_id
                and c.metadata.get("privacy_level") in ("S2", "S3")
            ),
        )

        return context[:max_entries]

    async def persist_conversation(
        self,
        session_id: str,
        entry: MemoryEntry,
        privacy_level: str,
    ) -> None:
        """Persist a conversation entry to the appropriate storage.

        This is separate from add() to allow persisting full conversation
        pairs in addition to individual messages.

        Args:
            session_id: session identifier.
            entry: conversation entry to persist.
            privacy_level: S1/S2/S3 - determines storage location.
        """
        if privacy_level == "S1":
            if self._cloud_memory is not None:
                await self._cloud_memory.add(entry)
                logger.debug("persisted_conversation_to_cloud", session_id=session_id)
        else:
            if self._local_memory is not None:
                await self._local_memory.add(entry)
                logger.debug("persisted_conversation_to_local", session_id=session_id)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "cached_sessions": len(self._cache),
            "total_entries": sum(
                s.total_messages for s in self._cache.values()
            ),
            "cloud_available": self._cloud_memory is not None,
            "local_available": self._local_memory is not None,
        }
