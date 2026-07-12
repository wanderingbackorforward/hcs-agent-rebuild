"""Tool result cache - caches retrieval/search results to avoid redundant computation.

When the same query is searched multiple times, return cached results
instead of re-running Hybrid Search. Invalidated on document updates.

Two backends:
- InMemoryToolCache: process-local dict store (default when Redis unavailable)
- RedisToolCache: Redis-backed with SETEX TTL (used when REDIS_URL is set)

The create_tool_cache() factory picks the right backend automatically.
All Redis errors degrade to a no-op (return None on get, skip on set)
so the application never crashes due to Redis being unavailable.
"""
import hashlib
import json
import time
import logging
from typing import Optional, Any

from config.settings import app_settings
from config.database import get_redis_client

logger = logging.getLogger(__name__)

DEFAULT_TTL = app_settings.tool_cache_ttl

# Import RedisError defensively so the module loads even without the redis
# package installed. When redis is missing, get_redis_client() returns None
# and RedisToolCache is never instantiated, so this fallback is harmless.
try:
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - redis is in requirements.txt
    RedisError = Exception


class InMemoryToolCache:
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


class RedisToolCache:
    """Redis-backed cache for tool/retrieval results.

    Keys:   tool_cache:{sha256_hash}
    Values: JSON-serialized result, stored with SETEX for automatic TTL.

    All Redis errors are caught and logged — the cache degrades to a no-op
    (returns None on get, skips on set) rather than crashing the caller.
    """

    KEY_PREFIX = "tool_cache:"

    def __init__(self, redis_client, ttl: int = DEFAULT_TTL):
        self.redis = redis_client
        self.ttl = ttl

    @staticmethod
    def _hash(query: str, tool_name: str, **kwargs) -> str:
        key_str = f"{tool_name}:{query}:{sorted(kwargs.items())}"
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _make_key(self, query: str, tool_name: str, **kwargs) -> str:
        return f"{self.KEY_PREFIX}{self._hash(query, tool_name, **kwargs)}"

    def get(self, query: str, tool_name: str, **kwargs) -> Optional[Any]:
        """Get cached tool result from Redis."""
        key = self._make_key(query, tool_name, **kwargs)
        try:
            raw = self.redis.get(key)
        except RedisError as e:
            logger.warning(f"Redis tool cache get failed: {e}")
            return None
        if raw is None:
            return None
        try:
            logger.info(f"Tool cache hit (Redis): {tool_name}:{query[:50]}")
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Redis tool cache deserialization failed: {e}")
            return None

    def set(self, query: str, tool_name: str, result: Any, **kwargs):
        """Cache a tool result in Redis with TTL."""
        key = self._make_key(query, tool_name, **kwargs)
        try:
            self.redis.setex(key, self.ttl, json.dumps(result))
        except RedisError as e:
            logger.warning(f"Redis tool cache set failed: {e}")
        except (TypeError, ValueError) as e:
            logger.warning(f"Redis tool cache serialization failed: {e}")

    def invalidate_all(self):
        """Delete all tool_cache:* keys from Redis via SCAN."""
        try:
            cursor = 0
            while True:
                cursor, keys = self.redis.scan(
                    cursor=cursor, match=f"{self.KEY_PREFIX}*"
                )
                if keys:
                    self.redis.delete(*keys)
                if int(cursor) == 0:
                    break
            logger.info("Tool cache cleared (Redis, document update)")
        except RedisError as e:
            logger.warning(f"Redis tool cache invalidate_all failed: {e}")


def create_tool_cache(ttl: Optional[int] = None):
    """Factory: return a Redis-backed cache if Redis is available, else in-memory.

    Args:
        ttl: Cache TTL in seconds. Defaults to app_settings.tool_cache_ttl.

    Returns:
        RedisToolCache if a Redis client is available, otherwise InMemoryToolCache.
    """
    if ttl is None:
        ttl = DEFAULT_TTL
    redis_client = get_redis_client()
    if redis_client is not None:
        return RedisToolCache(redis_client, ttl)
    return InMemoryToolCache(ttl)


# Backward-compat alias: existing code/tests do ``from cache.tool_cache import ToolCache``.
ToolCache = InMemoryToolCache
