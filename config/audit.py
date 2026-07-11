"""Full-chain audit logging with trace_id propagation.

Provides:
- TraceContext: async-safe trace_id context (contextvars)
- AuditLogger: structured JSON audit events (input, intent, tool_call, output)
- audit_event: convenience function to emit structured audit logs
- mask_sensitive: recursively mask sensitive dict fields
- sanitize_text: mask sensitive patterns (paths, keys, credentials) in text

trace_id is generated at BFF entry and propagated through
Orchestrator -> Worker -> Tool Server via contextvars.
"""
import json
import logging
import re
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

_audit_logger = logging.getLogger("audit")

# Async-safe trace context: each request gets its own trace_id
# that propagates through the entire call chain.
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_session_id: ContextVar[str] = ContextVar("session_id", default="")
_agent_name: ContextVar[str] = ContextVar("agent_name", default="")


def get_trace_id() -> str:
    """Return the current trace_id, or empty string if not set."""
    return _trace_id.get()


def get_session_id() -> str:
    return _session_id.get()


def get_agent_name() -> str:
    return _agent_name.get()


def set_trace_context(
    *,
    trace_id: Optional[str] = None,
    session_id: Optional[str] = None,
    agent_name: Optional[str] = None,
) -> None:
    """Set trace context values for the current async context.

    Any None argument leaves the existing value unchanged.
    """
    if trace_id is not None:
        _trace_id.set(trace_id)
    if session_id is not None:
        _session_id.set(session_id)
    if agent_name is not None:
        _agent_name.set(agent_name)


def new_trace_id() -> str:
    """Generate a new trace_id and set it in the current context."""
    tid = uuid.uuid4().hex[:16]
    _trace_id.set(tid)
    return tid


def clear_trace_context() -> None:
    """Reset all trace context vars to empty (for tests / cleanup)."""
    _trace_id.set("")
    _session_id.set("")
    _agent_name.set("")


@dataclass
class AuditEvent:
    """A single structured audit record spanning the full request chain."""

    layer: str  # "bff" | "orchestrator" | "worker" | "tool_server"
    event_type: str  # "request_in" | "intent_decision" | "tool_call" | "tool_result" | "response_out" | "error"
    trace_id: str = ""
    session_id: str = ""
    agent_name: str = ""
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "layer": self.layer,
            "agent": self.agent_name,
            "event": self.event_type,
            "message": self.message,
            "data": self.data,
        }


def audit_event(
    layer: str,
    event_type: str,
    *,
    message: str = "",
    data: Optional[dict[str, Any]] = None,
    level: int = logging.INFO,
) -> AuditEvent:
    """Emit a structured JSON audit log record.

    Automatically pulls trace_id / session_id / agent_name from context.
    Safe to call from any layer (BFF, Orchestrator, Worker, Tool Server).
    """
    event = AuditEvent(
        layer=layer,
        event_type=event_type,
        trace_id=get_trace_id(),
        session_id=get_session_id(),
        agent_name=get_agent_name(),
        message=message,
        data=data or {},
    )
    _audit_logger.log(level, json.dumps(event.to_dict(), ensure_ascii=False, default=str))
    return event


def mask_sensitive(value: Any, fields: Optional[set[str]] = None) -> Any:
    """Recursively mask sensitive fields in a dict for audit logging.

    Default sensitive fields: api_key, password, token, secret, authorization.
    Values are replaced with '***MASKED***'.
    """
    sensitive = fields or {"api_key", "password", "token", "secret", "authorization", "api_keys"}
    if isinstance(value, dict):
        return {
            k: ("***MASKED***" if k.lower() in sensitive else mask_sensitive(v, sensitive))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [mask_sensitive(v, sensitive) for v in value]
    return value


# ---------------------------------------------------------------------------
# sanitize_text: mask sensitive patterns in free-form text (error messages,
# log lines, tracebacks). Used for both stdout (returned to LLM) and stderr
# (logs inherited by parent process / Host).
# ---------------------------------------------------------------------------

# URL with embedded credentials: redis://user:password@host, postgres://...
_RE_URL_CREDS = re.compile(r'(\w+://[^\s:/@]+):([^\s/@]+)@', re.IGNORECASE)

# Windows absolute paths: C:\Users\..., D:\mine\...
_RE_WIN_PATH = re.compile(r'[A-Za-z]:\\[^\s\'"<>]+(?:\\[^\s\'"<>]+)+')

# Unix absolute paths under common root directories
_RE_UNIX_PATH = re.compile(
    r'/(?:home|usr|opt|etc|var|tmp|root|Users|mnt|data|app|srv)/[^\s\'"<>]+'
)

# API key patterns: sk-xxx, sk-proj-xxx, AKIAxxx, etc.
# Allows dashes in key body (e.g., sk-proj-abc123def456...).
_RE_API_KEY = re.compile(r'(?:sk|ak|AKIA)[-_]?[a-zA-Z0-9][a-zA-Z0-9\-]{14,}')

# Bearer tokens in headers
_RE_BEARER = re.compile(r'Bearer\s+[a-zA-Z0-9._\-]+', re.IGNORECASE)

# key=value or key: value patterns for known sensitive field names
_RE_KV_SECRET = re.compile(
    r'((?:api[_-]?key|token|password|passwd|secret|authorization|access[_-]?key)\s*[=:])\s*[^\s\'";,}>)]+',
    re.IGNORECASE,
)


def sanitize_text(text: str, *, max_len: int = 200) -> str:
    """Mask sensitive patterns in free-form text.

    Handles:
    - File paths (Windows / Unix) → <path>
    - API keys (sk-xxx, AKIAxxx) → <masked_key>
    - Bearer tokens → Bearer ***
    - URL credentials (redis://user:pass@host) → user:***@host
    - key=value secret patterns → key=***

    Truncates to *max_len* characters to prevent large payload leakage.
    """
    if not text:
        return text or ""
    result = str(text)
    result = _RE_URL_CREDS.sub(r'\1:***@', result)
    result = _RE_WIN_PATH.sub('<path>', result)
    result = _RE_UNIX_PATH.sub('<path>', result)
    result = _RE_API_KEY.sub('<masked_key>', result)
    result = _RE_BEARER.sub('Bearer ***', result)
    result = _RE_KV_SECRET.sub(r'\1***', result)
    if len(result) > max_len:
        result = result[:max_len] + "...(truncated)"
    return result
