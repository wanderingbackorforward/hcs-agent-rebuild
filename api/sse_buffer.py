"""Ring buffer for SSE event replay (Last-Event-ID support).

Keeps the last *capacity* events per session so that a client reconnecting
with ``Last-Event-ID`` header can replay missed events without data loss.

Lifecycle:
  1. ``append(session_id, event)`` on every emitted event.
  2. On reconnect, ``replay(session_id, last_seq)`` yields events after
     *last_seq*.
  3. ``prune()`` removes sessions whose TTL has expired (call periodically).
"""
import time
from collections import deque
from threading import Lock
from typing import Iterator

from api.sse_protocol import SSEEvent


class SSEBuffer:
    """Per-session ring buffer for SSE event replay."""

    def __init__(self, capacity: int = 50, session_ttl: float = 300):
        self._capacity = capacity
        self._ttl = session_ttl
        self._bufs: dict[str, deque] = {}
        self._last_seen: dict[str, float] = {}
        self._lock = Lock()

    def append(self, session_id: str, event: SSEEvent) -> None:
        with self._lock:
            buf = self._bufs.setdefault(
                session_id, deque(maxlen=self._capacity),
            )
            buf.append(event)
            self._last_seen[session_id] = time.time()

    def replay(self, session_id: str, last_seq: int) -> Iterator[SSEEvent]:
        with self._lock:
            buf = self._bufs.get(session_id)
            if not buf:
                return iter(())
            return iter(
                e for e in buf
                if e.seq > last_seq
            )

    def prune(self) -> int:
        """Remove expired sessions. Returns number removed."""
        now = time.time()
        expired = [
            sid for sid, ts in self._last_seen.items()
            if now - ts > self._ttl
        ]
        with self._lock:
            for sid in expired:
                self._bufs.pop(sid, None)
                self._last_seen.pop(sid, None)
        return len(expired)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._bufs.pop(session_id, None)
            self._last_seen.pop(session_id, None)


# Process-level singleton.
_sse_buffer: SSEBuffer | None = None


def get_sse_buffer() -> SSEBuffer:
    global _sse_buffer
    if _sse_buffer is None:
        _sse_buffer = SSEBuffer()
    return _sse_buffer
