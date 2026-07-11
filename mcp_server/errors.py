"""Shared MCP error types and formatting.

Dual-layer error mechanism:
  - Protocol-level errors: raise McpError(ErrorData(code=..., message=...))
    for JSON-RPC error responses (tool not found, permission denied).
  - Business-level errors: return CallToolResult(isError=True) with
    actionable hints telling the LLM what to do next.

Both stdout (error messages returned to LLM) and stderr (traceback logs)
are sanitized via sanitize_text() to strip file paths, API keys, and
other internal information.
"""
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Optional

from config.audit import sanitize_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default actionable hints per error_type.
# These tell the LLM what to do next when it receives a business-level error.
# ---------------------------------------------------------------------------
_DEFAULT_HINTS: dict[str, str] = {
    "not_found": "Try a different query, use list_collections to discover available documents, or verify the doc_id spelling.",
    "permission_denied": "This tool is restricted. Ask the user to check agent role configuration or use a different tool.",
    "invalid_input": "Check the parameter values against the tool schema and retry with valid inputs.",
    "timeout": "The operation timed out. Retry with a simpler query or try again later.",
    "network_error": "A network issue occurred. Retry the call, or check if the knowledge service is reachable.",
    "internal_error": "An unexpected error occurred. Retry with different parameters, or report the trace_id for investigation.",
}


def _default_hint(error_type: str) -> str:
    """Return the default actionable hint for an error_type."""
    return _DEFAULT_HINTS.get(error_type, "Retry with different parameters or contact support with the trace_id.")


class MCPError(Exception):
    """An expected MCP tool failure. Carries a stable error_type code.

    Attributes:
        error_type: Stable code for classification (not_found, invalid_input, etc.)
        message: Human-readable error description (safe for LLM).
        details: Optional structured details dict.
        hint: Actionable advice telling the LLM what to do next.
    """

    def __init__(
        self,
        error_type: str,
        message: str,
        *,
        details: Optional[dict] = None,
        hint: Optional[str] = None,
    ):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.details = details or {}
        self.hint = hint or _default_hint(error_type)


@dataclass
class FormattedError:
    """Sanitized error envelope returned to the LLM via CallToolResult."""

    error_type: str
    message: str
    trace_id: str
    details: Optional[dict] = None
    hint: str = ""

    def __post_init__(self):
        if not self.hint:
            self.hint = _default_hint(self.error_type)

    def to_text(self) -> str:
        """Compact, single-line, safe to send back to LLM. No internal info.

        Format: error_type=...; message=...; hint=...; trace_id=...
        The hint field tells the LLM what to do next.
        """
        parts = [f"error_type={self.error_type}", f"message={self.message}"]
        if self.details:
            parts.append(f"details={self.details}")
        parts.append(f"hint={self.hint}")
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

    If e is MCPError, use its declared type/message/hint.
    Otherwise, log full traceback (server-side, sanitized) and return a
    generic envelope with the exception message sanitized.
    """
    trace_id = uuid.uuid4().hex[:12]
    if isinstance(e, MCPError):
        return FormattedError(
            error_type=e.error_type,
            message=e.message,
            trace_id=trace_id,
            details=e.details,
            hint=e.hint,
        )
    # Unexpected exception: log full traceback (sanitized for stderr safety),
    # return generic envelope with sanitized message for stdout safety.
    logger.error(
        "unexpected exception in MCP tool [trace_id=%s context=%s]\n%s",
        trace_id, context, sanitize_text(traceback.format_exc(), max_len=2000),
    )
    return FormattedError(
        error_type=_classify_exception(e),
        message=sanitize_text(str(e)),
        trace_id=trace_id,
    )
