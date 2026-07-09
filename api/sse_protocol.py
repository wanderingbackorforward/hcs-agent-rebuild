"""Re-export from config.sse_protocol for backward compatibility.

The SSEEvent class and helpers live in config/sse_protocol.py to avoid
circular imports (agents layer needs SSEEvent, but api/__init__.py
eagerly imports chat_handler which imports agents).
"""
from config.sse_protocol import (  # noqa: F401
    SSEEvent,
    collect_text,
    format_sse_stream,
)

__all__ = ["SSEEvent", "collect_text", "format_sse_stream"]
