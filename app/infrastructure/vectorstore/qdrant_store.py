"""Qdrant-backed vector store for long-term memory and RAG."""

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.exceptions.exceptions import MemoryException
from app.core.logger.logger import get_logger
from app.domain.memory.memory import MemoryEntry, MemoryStore, MemoryType

logger = get_logger(__name__)


class QdrantMemoryStore(MemoryStore):
    """Long-term memory backed by Qdrant vector database."""

    memory_type = MemoryType.LONG_TERM

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection: str = "agent_memory",
        vector_size: int = 1536,
    ) -> None:
        self._host = host
        self._port = port
        self._collection = collection
        self._vector_size = vector_size
        self._client: AsyncQdrantClient | None = None

    async def _ensure_client(self) -> AsyncQdrantClient:
        """Lazily create the Qdrant client."""
        if self._client is None:
            self._client = AsyncQdrantClient(host=self._host, port=self._port)
            # Create collection if it doesn't exist
            collections = await self._client.get_collections()
            names = [c.name for c in collections.collections]
            if self._collection not in names:
                await self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=self._vector_size, distance=Distance.COSINE
                    ),
                )
                logger.info("qdrant_collection_created", collection=self._collection)
        return self._client

    async def add(self, entry: MemoryEntry) -> str:
        """Store a memory entry with its embedding."""
        raise NotImplementedError("Embedding computation must be provided externally")

    async def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search by vector similarity (requires external embedding)."""
        raise NotImplementedError("Embedding computation must be provided externally")

    async def get_recent(self, limit: int = 10) -> list[MemoryEntry]:
        """Get most recent entries (scroll by insertion order)."""
        client = await self._ensure_client()
        try:
            results = await client.scroll(
                collection_name=self._collection,
                limit=limit,
                with_payload=True,
            )
            return [
                MemoryEntry(
                    content=point.payload.get("content", ""),
                    metadata=point.payload.get("metadata", {}),
                    session_id=point.payload.get("session_id", ""),
                    entry_id=str(point.id),
                )
                for point in results[0]
            ]
        except Exception as exc:
            raise MemoryException(f"Failed to get recent memories: {exc}") from exc

    async def clear(self, session_id: str | None = None) -> None:
        """Clear all memories (optionally filter by session)."""
        client = await self._ensure_client()
        try:
            if session_id:
                await client.delete(
                    collection_name=self._collection,
                    points_selector={
                        "filter": {
                            "must": [
                                {"key": "session_id", "match": {"value": session_id}}
                            ]
                        }
                    },
                )
            else:
                await client.delete_collection(self._collection)
                logger.info("qdrant_collection_deleted", collection=self._collection)
        except Exception as exc:
            raise MemoryException(f"Failed to clear memories: {exc}") from exc
