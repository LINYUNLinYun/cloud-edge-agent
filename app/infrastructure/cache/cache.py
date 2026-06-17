"""Simple in-memory cache for repeated queries and embeddings.

Replace with Redis-backed implementation for production.
"""

import hashlib
import time
from dataclasses import dataclass

from app.core.logger.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """A cached item with TTL."""

    value: object
    expires_at: float


class InMemoryCache:
    """Simple TTL-based in-memory cache."""

    def __init__(self, default_ttl: float = 300.0) -> None:
        self._store: dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl

    def _make_key(self, namespace: str, key: str) -> str:
        """Build a namespaced cache key."""
        raw = f"{namespace}:{key}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, namespace: str, key: str) -> object | None:
        """Retrieve a cached value, or None if missing/expired."""
        full_key = self._make_key(namespace, key)
        entry = self._store.get(full_key)
        if entry is None:
            return None
        if time.monotonic() > entry.expires_at:
            del self._store[full_key]
            return None
        return entry.value

    def set(
        self, namespace: str, key: str, value: object, ttl: float | None = None
    ) -> None:
        """Store a value in the cache."""
        full_key = self._make_key(namespace, key)
        self._store[full_key] = CacheEntry(
            value=value,
            expires_at=time.monotonic() + (ttl or self._default_ttl),
        )

    def invalidate(self, namespace: str, key: str) -> None:
        """Remove a specific cache entry."""
        full_key = self._make_key(namespace, key)
        self._store.pop(full_key, None)

    def clear(self) -> None:
        """Clear all cached data."""
        self._store.clear()
