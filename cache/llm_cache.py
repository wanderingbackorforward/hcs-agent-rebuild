"""LLM result cache - query hash to response with TTL.

Caches LLM responses by query hash to avoid redundant API calls for
identical questions. TTL ensures stale entries are evicted automatically.
"""
import hashlib
import time
import logging
from typing import Optional, Any

from config.settings import app_settings

logger = logging.getLogger(__name__)

DEFAULT_TTL = app_settings.llm_cache_ttl


class LLMCache:
    """Simple in-memory LLM result cache with TTL."""

    def __init__(self, ttl: int = DEFAULT_TTL):
        self.ttl = ttl
        self._store: dict[str, tuple[Any, float]] = {}

    @staticmethod
    def _hash(query: str, **kwargs) -> str:
        """Generate cache key from query and relevant params."""
        key_str = f"{query}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, query: str, **kwargs) -> Optional[str]:
        """Get cached response if not expired."""
        key = self._hash(query, **kwargs)
        if key in self._store:
            response, timestamp = self._store[key]
            if time.time() - timestamp < self.ttl:
                logger.info(f"LLM cache hit: {query[:50]}")
                return response
            else:
                del self._store[key]
        return None

    def set(self, query: str, response: str, **kwargs):
        """Cache a response."""
        key = self._hash(query, **kwargs)
        self._store[key] = (response, time.time())
        logger.info(f"LLM cache set: {query[:50]}")

    def invalidate(self, query: str = None, **kwargs):
        """Invalidate specific entry or all entries."""
        if query:
            key = self._hash(query, **kwargs)
            self._store.pop(key, None)
        else:
            self._store.clear()

    def stats(self) -> dict:
        """Return cache statistics."""
        now = time.time()
        active = sum(1 for _, ts in self._store.values() if now - ts < self.ttl)
        expired = len(self._store) - active
        return {"total": len(self._store), "active": active, "expired": expired}
