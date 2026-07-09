"""BFF middleware: trace_id injection + structured audit logging.

Uses pure ASGI middleware (not BaseHTTPMiddleware) to avoid
known incompatibilities with FastAPI validation error responses
and streaming responses.

Generates a trace_id at request entry, stores it in the async-safe
ContextVar so it propagates through Orchestrator -> Worker -> Tool
Server automatically. Adds X-Trace-Id to the response header.
"""
import logging
import time

from starlette.types import ASGIApp, Receive, Scope, Send

from config.audit import audit_event, new_trace_id, set_trace_context

logger = logging.getLogger(__name__)


class TraceIdMiddleware:
    """Pure ASGI middleware: injects trace_id into every request.

    - Reads X-Trace-Id header if present (for upstream propagation).
    - Otherwise generates a new trace_id.
    - Sets trace_id in ContextVar for the request lifetime.
    - Injects X-Trace-Id into the response headers.
    - Emits structured audit events at request_in and response_out.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        if scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        # Extract or generate trace_id from request headers.
        headers = dict(scope.get("headers", []))
        trace_id = headers.get(b"x-trace-id", b"").decode() or new_trace_id()
        session_id = headers.get(b"x-session-id", b"").decode()

        set_trace_context(trace_id=trace_id, session_id=session_id)

        path = scope.get("path", "")
        method = scope.get("method", "")
        client = scope.get("client", ["", 0])[0] if scope.get("client") else None

        audit_event(
            layer="bff",
            event_type="request_in",
            message=f"{method} {path}",
            data={"method": method, "path": path, "client": client},
        )

        start = time.monotonic()
        status_code = 0

        async def send_wrapper(message):
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
                # Inject X-Trace-Id into response headers.
                raw_headers = list(message.get("headers", []))
                raw_headers.append((b"x-trace-id", trace_id.encode()))
                message = {**message, "headers": raw_headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as e:
            elapsed_ms = round((time.monotonic() - start) * 1000, 2)
            audit_event(
                layer="bff",
                event_type="error",
                message=f"Unhandled exception: {type(e).__name__}",
                data={"elapsed_ms": elapsed_ms, "error": str(e)[:200]},
                level=logging.ERROR,
            )
            raise

        elapsed_ms = round((time.monotonic() - start) * 1000, 2)
        audit_event(
            layer="bff",
            event_type="response_out",
            message=f"{method} {path} -> {status_code}",
            data={"status_code": status_code, "elapsed_ms": elapsed_ms},
        )
