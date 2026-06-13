"""Shared MCP error types and formatting.

MCP tool handlers MUST NOT leak raw exceptions back to the LLM/Host.
Use MCPError for expected failures; let unexpected ones still log via
logger.exception but return a sanitized message + trace_id.
"""
import logging
import traceback
import uuid
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


class MCPError(Exception):
    """An expected MCP tool failure. Carries a stable error_type code."""

    def __init__(self, error_type: str, message: str, *, details: Optional[dict] = None):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = details or {}


@dataclass
class FormattedError:
    error_type: str
    message: str
    trace_id: str
    details: Optional[dict] = None

    def to_text(self) -> str:
        # Compact, single-line, safe to send back to LLM. No internal info.
        parts = [f"error_type={self.error_type}", f"message={self.message}"]
        if self.details:
            parts.append(f"details={self.details}")
        parts.append(f"trace_id={self.trace_id}")
        return "; ".join(parts)


def _classify_exception(e: Exception) -> str:
    """Map common exception classes to a stable error_type code."""
    name = type(e).__name__
    if "NotFound" in name or "Missing" in name:
        return "not_found"
    if "Permission" in name or "Auth" in name:
        return "permission_denied"
    if "Validation" in name or "ValueError" in name:
        return "invalid_input"
    if "Timeout" in name:
        return "timeout"
    if "Connection" in name or "Network" in name:
        return "network_error"
    return "internal_error"


def format_error(e: Exception, *, context: str = "") -> FormattedError:
    """Build a sanitized error envelope from an exception.

    If e is MCPError, use its declared type/message.
    Otherwise, log full traceback (server-side) and return a generic envelope.
    """
    trace_id = uuid.uuid4().hex[:12]
    if isinstance(e, MCPError):
        return FormattedError(
            error_type=e.error_type,
            message=e.message,
            trace_id=trace_id,
            details=e.details,
        )
    # Unexpected exception: log full traceback, return generic envelope.
    logger.error(
        "unexpected exception in MCP tool [trace_id=%s context=%s]\n%s",
        trace_id, context, traceback.format_exc(),
    )
    return FormattedError(
        error_type=_classify_exception(e),
        message=str(e) if len(str(e)) < 200 else f"{type(e).__name__} (truncated)",
        trace_id=trace_id,
    )
