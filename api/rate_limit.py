"""Per-key sliding-window rate limiter.

Supports two backends:
- In-process (default): lightweight, no external deps. Single-process only.
- Redis (optional): set REDIS_URL to enable. Shared across multiple workers.

Default: 10 requests / 60 seconds per key. Configurable via env.
"""
import logging
import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import HTTPException, Request, status

from config.database import redis_config

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
DEFAULT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX", "10"))


class InProcessLimiter:
    """In-process sliding-window limiter. Single-process only."""

    def __init__(self, max_requests: int = DEFAULT_MAX_REQUESTS, window_seconds: int = DEFAULT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    async def check(self, key: str) -> None:
        """Raises 429 if the key has exceeded max_requests in the window."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        dq = self._hits[key]
        # Drop expired entries from the left.
        while dq and dq[0] < cutoff:
            dq.popleft()
        if len(dq) >= self.max_requests:
            retry_after = int(self.window_seconds - (now - dq[0])) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.max_requests} req / {self.window_seconds}s. "
                       f"Retry after {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )
        dq.append(now)


class RedisLimiter:
    """Redis-backed sliding-window limiter. Supports multi-process deployment.

    Uses a sorted set per key: members are timestamps, scores are timestamps.
    On each request, ZRANGEBYSCORE removes expired entries, then ZCARD checks
    the count. If under limit, ZADD appends the current timestamp with an
    auto-expiring TTL so stale keys don't leak memory.
    """

    def __init__(self, max_requests: int = DEFAULT_MAX_REQUESTS, window_seconds: int = DEFAULT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._redis = None

    async def _get_redis(self):
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(redis_config.url, decode_responses=True)
        return self._redis

    async def check(self, key: str) -> None:
        """Raises 429 if the key has exceeded max_requests in the window."""
        r = await self._get_redis()
        now = time.time()
        cutoff = now - self.window_seconds
        pipe = r.pipeline()
        # Remove expired entries.
        pipe.zremrangebyscore(f"rl:{key}", 0, cutoff)
        # Count remaining entries.
        pipe.zcard(f"rl:{key}")
        # Add current request.
        pipe.zadd(f"rl:{key}", {str(now): now})
        # Auto-expire the key after window seconds to prevent memory leak.
        pipe.expire(f"rl:{key}", self.window_seconds + 1)
        results = await pipe.execute()
        count = results[1]  # ZCARD result (after ZREMRANGEBYSCORE)
        if count >= self.max_requests:
            # The oldest entry in the set is the earliest timestamp.
            oldest = await r.zrange(f"rl:{key}", 0, 0, withscores=True)
            if oldest:
                retry_after = int(self.window_seconds - (now - oldest[0][1])) + 1
            else:
                retry_after = self.window_seconds
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {self.max_requests} req / {self.window_seconds}s. "
                       f"Retry after {retry_after}s.",
                headers={"Retry-After": str(retry_after)},
            )


def _create_limiter():
    """Factory: returns RedisLimiter if REDIS_URL is configured, otherwise InProcessLimiter."""
    if redis_config.enabled:
        logger.info("Redis URL configured, using Redis-backed rate limiter")
        return RedisLimiter()
    logger.info("No Redis URL, using in-process rate limiter")
    return InProcessLimiter()


_limiter = _create_limiter()


async def rate_limit(request: Request) -> None:
    """FastAPI dependency. Reads the api_key that require_api_key stored on
    request.state; falls back to "anonymous" in dev mode."""
    api_key = getattr(request.state, "api_key", None) or "anonymous"
    await _limiter.check(api_key)
