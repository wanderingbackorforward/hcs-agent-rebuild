"""In-process per-key sliding-window rate limiter.

Lightweight alternative to slowapi: ~50 lines, no external dep.
Default: 10 requests / 60 seconds per key. Configurable via env.

In a multi-process deployment, swap this for a Redis-backed limiter.
"""
import logging
import os
import time
from collections import defaultdict, deque
from typing import Deque, Dict

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW", "60"))
DEFAULT_MAX_REQUESTS = int(os.getenv("RATE_LIMIT_MAX", "10"))


class SlidingWindowLimiter:
    def __init__(self, max_requests: int = DEFAULT_MAX_REQUESTS, window_seconds: int = DEFAULT_WINDOW_SECONDS):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)

    def check(self, key: str) -> None:
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


_limiter = SlidingWindowLimiter()


def rate_limit(api_key: str) -> None:
    """FastAPI dependency wrapper."""
    _limiter.check(api_key)
