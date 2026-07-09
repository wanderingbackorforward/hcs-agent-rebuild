"""SSE event protocol for structured streaming.

Defines SSEEvent — the single event type that flows through the agent
pipeline.  Agents may yield plain ``str`` (treated as token content for
backward compatibility) or ``SSEEvent`` for structured events.

Event types:
  status    — processing stage update (e.g. "正在分类意图...")
  token     — reply content chunk
  decision  — routing decision metadata
  error     — error with retry info
  done      — completion signal
"""
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator


@dataclass
class SSEEvent:
    """One structured event in the SSE stream."""

    type: str                          # status | token | decision | error | done
    data: dict[str, Any] = field(default_factory=dict)
    seq: int = 0                       # set by the transport layer

    # -- factories --------------------------------------------------

    @classmethod
    def status(cls, stage: str, message: str) -> "SSEEvent":
        return cls(type="status", data={
            "stage": stage, "message": message, "timestamp": time.time(),
        })

    @classmethod
    def token(cls, content: str) -> "SSEEvent":
        return cls(type="token", data={"content": content})

    @classmethod
    def decision(cls, **kwargs: Any) -> "SSEEvent":
        return cls(type="decision", data=dict(kwargs))

    @classmethod
    def error(cls, error_type: str, message: str,
              retryable: bool = False, suggestion: str = "") -> "SSEEvent":
        return cls(type="error", data={
            "error_type": error_type, "message": message,
            "retryable": retryable, "suggestion": suggestion,
        })

    @classmethod
    def done(cls, **kwargs: Any) -> "SSEEvent":
        return cls(type="done", data=dict(kwargs))

    # -- helpers ----------------------------------------------------

    @property
    def text(self) -> str:
        """Return token content if this is a token event, else empty."""
        return self.data.get("content", "") if self.type == "token" else ""

    def to_sse(self) -> str:
        """Format as SSE wire text: ``event: {type}\\ndata: {json}\\n\\n``."""
        payload = json.dumps(self.data, ensure_ascii=False, default=str)
        return f"event: {self.type}\ndata: {payload}\n\n"

    def __str__(self) -> str:
        """Plain-text fallback (for non-SSE consumers)."""
        return self.text


async def collect_text(stream: AsyncGenerator) -> str:
    """Accumulate all token events from *stream* into a single string.

    Used by the non-streaming ``/chat`` endpoint to collect the full reply
    while still processing status/decision events (which are silently
    consumed).
    """
    result = ""
    async for item in stream:
        if isinstance(item, SSEEvent):
            result += item.text
        else:
            result += str(item)
    return result


async def format_sse_stream(
    stream: AsyncGenerator,
    session_id: str = "",
    start_seq: int = 0,
):
    """Convert a mixed ``str``/``SSEEvent`` stream into SSE wire text.

    * Assigns monotonically increasing ``seq`` to every event.
    * Plain strings are wrapped as ``token`` events.
    * If *session_id* is set, events are appended to the global SSE
      buffer for Last-Event-ID replay.
    """
    from api.sse_buffer import get_sse_buffer
    buf = get_sse_buffer() if session_id else None
    seq = start_seq

    async for item in stream:
        if isinstance(item, SSEEvent):
            event = item
        else:
            event = SSEEvent.token(str(item))
        event.seq = seq
        if buf:
            buf.append(session_id, event)
        yield event.to_sse()
        seq += 1
