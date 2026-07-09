"""Task manager for cooperative cancellation and checkpoint resume.

Tracks active tasks by task_id using asyncio.Event for cooperative
cancellation. The streaming generator periodically checks
``is_cancelled(task_id)`` and exits gracefully, saving a checkpoint
that can be used for resume.

Design:
  * Cooperative (not preemptive): generators check is_cancelled() at
    natural yield points — no need for asyncio.Task wrapping.
  * In-memory only: checkpoints are lost on process restart (acceptable
    for MVP; session state in SQLite survives).
  * TTL 300s: auto-cleanup prevents unbounded memory growth.
"""
import asyncio
import time
from typing import Any

_TTL = 300  # 5 minutes


class TaskManager:
    """Process-level singleton for task tracking."""

    def __init__(self, ttl: float = _TTL):
        self._ttl = ttl
        self._events: dict[str, asyncio.Event] = {}
        self._checkpoints: dict[str, dict] = {}
        self._timestamps: dict[str, float] = {}
        self._lock = asyncio.Lock()

    def register(self, task_id: str) -> asyncio.Event:
        """Register a new task and return its cancellation event."""
        ev = asyncio.Event()
        self._events[task_id] = ev
        self._timestamps[task_id] = time.time()
        return ev

    def cancel(self, task_id: str) -> bool:
        """Signal cancellation for a task. Returns True if found."""
        ev = self._events.get(task_id)
        if ev is None:
            return False
        ev.set()
        return True

    def is_cancelled(self, task_id: str) -> bool:
        """Check if a task has been cancelled."""
        ev = self._events.get(task_id)
        return ev is not None and ev.is_set()

    def checkpoint(self, task_id: str, state: dict[str, Any]) -> None:
        """Save checkpoint state for potential resume."""
        self._checkpoints[task_id] = state

    def get_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        """Retrieve checkpoint state, or None if not found."""
        return self._checkpoints.get(task_id)

    def cleanup(self, task_id: str) -> None:
        """Remove all traces of a task."""
        self._events.pop(task_id, None)
        self._checkpoints.pop(task_id, None)
        self._timestamps.pop(task_id, None)

    def prune(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        expired = [
            tid for tid, ts in self._timestamps.items()
            if now - ts > self._ttl
        ]
        for tid in expired:
            self.cleanup(tid)
        return len(expired)


# Process-level singleton.
_task_manager: TaskManager | None = None


def get_task_manager() -> TaskManager:
    global _task_manager
    if _task_manager is None:
        _task_manager = TaskManager()
    return _task_manager


def make_task_id(session_id: str) -> str:
    """Generate a traceable task_id from session_id + nanosecond timestamp."""
    return f"{session_id}:{time.time_ns():x}"
