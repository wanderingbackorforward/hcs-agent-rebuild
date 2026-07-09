"""Tool result cache - caches retrieval/search results to avoid redundant computation.

When the same query is searched multiple times, return cached results
instead of re-running Hybrid Search. Invalidated on document updates.
"""
import hashlib
import time
import logging
from typing import Optional, List, Tuple, Dict, Any

from config.settings import app_settings

logger = logging.getLogger(__name__)

DEFAULT_TTL = app_settings.tool_cache_ttl


class ToolCache:
    """In-memory cache for tool/retrieval results."""

    def __init__(self, ttl: int = DEFAULT_TTL):
        self.ttl = ttl
        self._store: dict[str, tuple[Any, float]] = {}

    @staticmethod
    def _hash(query: str, tool_name: str, **kwargs) -> str:
        key_str = f"{tool_name}:{query}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def get(self, query: str, tool_name: str, **kwargs) -> Optional[Any]:
        """Get cached tool result if not expired."""
        key = self._hash(query, tool_name, **kwargs)
        if key in self._store:
            result, timestamp = self._store[key]
            if time.time() - timestamp < self.ttl:
                logger.info(f"Tool cache hit: {tool_name}:{query[:50]}")
                return result
            else:
                del self._store[key]
        return None

    def set(self, query: str, tool_name: str, result: Any, **kwargs):
        """Cache a tool result."""
        key = self._hash(query, tool_name, **kwargs)
        self._store[key] = (result, time.time())

    def invalidate_all(self):
        """Invalidate all tool cache entries (e.g., after document update)."""
        self._store.clear()
        logger.info("Tool cache cleared (document update)")
