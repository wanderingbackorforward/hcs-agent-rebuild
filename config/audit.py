"""Full-chain audit logging with trace_id propagation.

Provides:
- TraceContext: async-safe trace_id context (contextvars)
- AuditLogger: structured JSON audit events (input, intent, tool_call, output)
- audit_event: convenience function to emit structured audit logs

trace_id is generated at BFF entry and propagated through
Orchestrator -> Worker -> Tool Server via contextvars.
"""
import json
import logging
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
