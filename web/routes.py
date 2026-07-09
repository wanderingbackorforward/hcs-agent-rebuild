"""Web routes for HCS Agent Platform."""
import re
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from api.auth import require_api_key
from api.chat_handler import process_user_input_stream
from api.rate_limit import rate_limit
from config.audit import audit_event, mask_sensitive, set_trace_context

router = APIRouter(tags=["Web界面"])
templates = Jinja2Templates(directory="web/templates")

MAX_MESSAGE_LENGTH = 2000  # hard cap on a single user turn
MAX_SESSION_ID_LENGTH = 64

# Patterns that strongly indicate prompt-injection attempts. Conservative: only
# flag obvious overrides ("ignore previous", "system:", etc.). The classifier
# will catch what it can; this is a backstop, not a guarantee.
_INJECTION_PATTERNS = re.compile(
    r"(?i)(\bignore (?:all )?(?:previous|prior|above) (?:instructions?|prompts?)\b"
    r"|\bdisregard (?:all )?(?:previous|prior|above)\b"
    r"|^system:\s|^\s*<<\s*sys\s*>>|###\s*system\s*###|forget (?:everything|all))"
)


class ChatRequest(BaseModel):
    message: Annotated[str, Field(min_length=1, max_length=MAX_MESSAGE_LENGTH)]
    session_id: Annotated[str | None, Field(default=None, max_length=MAX_SESSION_ID_LENGTH)]

    @field_validator("message")
    @classmethod
    def _no_injection(cls, v: str) -> str:
        if _INJECTION_PATTERNS.search(v):
            raise ValueError(
                "Input contains patterns that look like prompt-injection "
                "attempts. Rephrase without instructing the model to ignore "
                "prior instructions."
            )
        return v.strip()

    @field_validator("session_id")
    @classmethod
    def _session_id_safe(cls, v: str | None) -> str | None:
        if v is None:
            return None
        # Restrict to URL-safe characters; prevents weird session IDs
        # leaking into file paths / DB keys / log lines.
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", v):
            raise ValueError("session_id must match [A-Za-z0-9_-]+")
        return v


def _validate_chat(chat: ChatRequest) -> str:
    """Normalize and re-check the chat request after Pydantic validation.

    Returns the message; raises 400 with a sanitized error if anything is off.
    """
    msg = chat.message.strip()
    if not msg:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="message must not be empty after stripping whitespace",
        )
    if len(msg) > MAX_MESSAGE_LENGTH:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"message exceeds {MAX_MESSAGE_LENGTH} characters",
        )
    return msg


@router.get("/", response_class=HTMLResponse, summary="主页")
async def read_root(request: Request):
    """Render chat home page."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.post(
    "/chat/stream",
    summary="流式聊天 (SSE)",
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
async def chat_stream_endpoint(chat: ChatRequest, request: Request):
    """SSE streaming endpoint — emits structured events."""
    msg = _validate_chat(chat)
    sid = chat.session_id or str(uuid.uuid4())
    set_trace_context(session_id=sid)
    audit_event(layer="bff", event_type="chat_request",
                message="sse streaming chat",
                data={"session_id": sid, "message_length": len(msg)})

    # Replay missed events if client sends Last-Event-ID header.
    from api.sse_buffer import get_sse_buffer
    from config.sse_protocol import SSEEvent, format_sse_stream
    buf = get_sse_buffer()
    last_seq = 0
    leid = request.headers.get("last-event-id", "")
    if leid.isdigit():
        last_seq = int(leid)

    async def event_generator():
        # Replay buffered events on reconnect.
        if last_seq:
            for e in buf.replay(sid, last_seq):
                yield e.to_sse()
        # Stream new events from the agent pipeline.
        try:
            async for chunk in format_sse_stream(
                process_user_input_stream(msg, session_id=sid),
                session_id=sid, start_seq=last_seq,
            ):
                yield chunk
            yield SSEEvent.done(session_id=sid).to_sse()
        except Exception as e:
            audit_event(layer="bff", event_type="error",
                        message=f"stream error: {type(e).__name__}",
                        data={"error": str(e)[:200]}, level=40)
            yield SSEEvent.error(
                type(e).__name__, "处理异常，请重试", retryable=True,
            ).to_sse()

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
    }
    return StreamingResponse(
        event_generator(), media_type="text/event-stream",
        headers=headers,
    )


@router.post(
    "/chat",
    summary="普通聊天接口",
    dependencies=[Depends(require_api_key), Depends(rate_limit)],
)
async def chat_endpoint(chat: ChatRequest):
    """Non-streaming chat endpoint (accumulates SSE stream to text)."""
    from config.sse_protocol import collect_text
    msg = _validate_chat(chat)
    sid = chat.session_id or str(uuid.uuid4())
    set_trace_context(session_id=sid)
    audit_event(layer="bff", event_type="chat_request",
                message="non-streaming chat",
                data={"session_id": sid, "message_length": len(msg)})
    result = await collect_text(
        process_user_input_stream(msg, session_id=sid)
    )
    audit_event(layer="bff", event_type="chat_response",
                message="non-streaming chat done",
                data={"session_id": sid, "reply_length": len(result)})
    return {"reply": result, "session_id": sid}
