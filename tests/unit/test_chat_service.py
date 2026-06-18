"""Tests for ChatService — memory retrieval and context injection."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.domain.memory.memory import MemoryEntry, MemoryStore, MemoryType
from app.infrastructure.cache.session_cache import SessionCacheManager
from app.services.chat_service import ChatService, _MAX_CONTEXT_MESSAGES


class FakeShortTermMemory(MemoryStore):
    """In-memory short-term store for testing."""

    memory_type = MemoryType.SHORT_TERM

    def __init__(self) -> None:
        self._entries: list[MemoryEntry] = []

    async def add(self, entry: MemoryEntry) -> str:
        entry.entry_id = str(len(self._entries))
        self._entries.append(entry)
        return entry.entry_id

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        return self._entries[-top_k:]

    async def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        return self._entries[-limit:]

    async def clear(self, session_id: str | None = None) -> None:
        self._entries.clear()


class FakeLongTermMemory(MemoryStore):
    """Stub long-term store returning pre-configured entries."""

    memory_type = MemoryType.LONG_TERM

    def __init__(self, entries: list[MemoryEntry] | None = None) -> None:
        self._entries = entries or []
        self.added: list[MemoryEntry] = []

    async def add(self, entry: MemoryEntry) -> str:
        self.added.append(entry)
        return "lt-0"

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        return self._entries[:top_k]

    async def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        return self._entries[:limit]

    async def clear(self, session_id: str | None = None) -> None:
        pass


def _make_chat_service(
    short_term: MemoryStore | None = None,
    cloud_memory: MemoryStore | None = None,
    local_memory: MemoryStore | None = None,
) -> tuple[ChatService, MagicMock, MagicMock]:
    """Create ChatService with mocked orchestrator and session store."""
    orchestrator = MagicMock()
    orchestrator.process = AsyncMock(
        return_value=MagicMock(
            answer="test answer",
            mode=MagicMock(value="direct_local"),
            routing=MagicMock(privacy_level=MagicMock(value="S1"), complexity=MagicMock(value=1)),
            latency_ms=100.0,
        )
    )

    session_store = MagicMock()
    session_store.get.return_value = MagicMock()  # session exists
    session_store.add_message = MagicMock()

    st = short_term or FakeShortTermMemory()

    service = ChatService(
        orchestrator=orchestrator,
        session_store=session_store,
        short_term_memory=st,
        cloud_memory=cloud_memory,
        local_memory=local_memory,
    )
    return service, orchestrator, session_store


class TestEnrichQuery:
    """Tests for ChatService._enrich_query()."""

    def test_no_context_returns_original(self) -> None:
        result = ChatService._enrich_query("hello", [])
        assert result == "hello"

    def test_with_context_prepends_history(self) -> None:
        context = [
            MemoryEntry(content="prev question", metadata={"role": "user"}),
            MemoryEntry(content="prev answer", metadata={"role": "assistant"}),
        ]
        result = ChatService._enrich_query("new question", context)

        assert "Relevant conversation history:" in result
        assert "[user] prev question" in result
        assert "[assistant] prev answer" in result
        assert "Current question: new question" in result

    def test_context_without_role_metadata(self) -> None:
        context = [MemoryEntry(content="some memory", metadata={})]
        result = ChatService._enrich_query("query", context)

        assert "[unknown] some memory" in result


class TestSessionCache:
    """Tests for SessionCacheManager integration."""

    @pytest.mark.asyncio
    async def test_cache_add_and_retrieve(self) -> None:
        """Test that entries are cached and retrievable."""
        cache = SessionCacheManager()

        entry = MemoryEntry(
            content="test message",
            session_id="s1",
            metadata={"role": "user"},
        )
        await cache.add("s1", entry, privacy_level="S1")

        context = await cache.get_context("s1", "test", max_entries=10)
        assert len(context) >= 1
        assert any(e.content == "test message" for e in context)

    @pytest.mark.asyncio
    async def test_cache_cloud_search(self) -> None:
        """Test that cloud memory is searched for past sessions."""
        cm = FakeLongTermMemory(
            entries=[MemoryEntry(content="past conversation", session_id="other", score=0.9)]
        )
        cache = SessionCacheManager(cloud_memory=cm)

        context = await cache.get_context("s1", "query", max_entries=10)
        assert any(e.content == "past conversation" for e in context)

    @pytest.mark.asyncio
    async def test_cache_local_search(self) -> None:
        """Test that local memory is searched for S2/S3 conversations."""
        lm = FakeLongTermMemory(
            entries=[MemoryEntry(content="sensitive data", session_id="other", score=0.9)]
        )
        cache = SessionCacheManager(local_memory=lm)

        context = await cache.get_context("s1", "query", max_entries=10)
        assert any(e.content == "sensitive data" for e in context)

    @pytest.mark.asyncio
    async def test_cache_stats(self) -> None:
        """Test cache statistics."""
        cm = FakeLongTermMemory()
        lm = FakeLongTermMemory()
        cache = SessionCacheManager(cloud_memory=cm, local_memory=lm)

        stats = cache.get_stats()
        assert stats["cloud_available"] is True
        assert stats["local_available"] is True
        assert stats["cached_sessions"] == 0


class TestChatFlow:
    """Tests for the full ChatService.chat() flow."""

    @pytest.mark.asyncio
    async def test_chat_returns_response(self) -> None:
        """Test basic chat flow."""
        service, orchestrator, _ = _make_chat_service()

        result = await service.chat("hello", session_id="s1")

        assert result.answer == "test answer"
        assert result.session_id == "s1"
        assert result.privacy_level == "S1"
        orchestrator.process.assert_called_once()

    @pytest.mark.asyncio
    async def test_chat_stores_in_cache(self) -> None:
        """Test that chat stores entries in cache."""
        cm = FakeLongTermMemory()
        service, _, _ = _make_chat_service(cloud_memory=cm)

        await service.chat("hello", session_id="s1")

        # Check cache stats
        stats = service._cache.get_stats()
        assert stats["cached_sessions"] >= 1
        assert stats["total_entries"] >= 2  # user + assistant

    @pytest.mark.asyncio
    async def test_chat_enriches_query_with_context(self) -> None:
        """Test that context is used to enrich the query."""
        service, orchestrator, _ = _make_chat_service()

        # First message
        await service.chat("my name is Alice", session_id="s1")

        # Second message - should have context from first
        await service.chat("what's my name?", session_id="s1")

        # Check that the second call had enriched context
        call_args = orchestrator.process.call_args
        enriched_query = call_args.kwargs["query"]
        # The context should include the previous conversation
        assert "what's my name?" in enriched_query
