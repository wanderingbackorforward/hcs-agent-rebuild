"""Web routes for HCS Agent Platform."""
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from api.chat_handler import process_user_input_stream

router = APIRouter(tags=["Web界面"])
templates = Jinja2Templates(directory="web/templates")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


@router.get("/", response_class=HTMLResponse, summary="主页")
async def read_root(request: Request):
    """Render chat home page."""
    return templates.TemplateResponse("index.html", {"request": request})


@router.post("/chat/stream", summary="流式聊天")
async def chat_stream_endpoint(chat: ChatRequest):
    """Handle streaming chat requests."""
    sid = chat.session_id or str(uuid.uuid4())

    async def token_generator():
        async for token in process_user_input_stream(chat.message, session_id=sid):
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain")


@router.post("/chat", summary="普通聊天接口")
async def chat_endpoint(chat: ChatRequest):
    """Non-streaming chat endpoint."""
    sid = chat.session_id or str(uuid.uuid4())
    result = ""
    async for token in process_user_input_stream(chat.message, session_id=sid):
        result += token
    return {"reply": result, "session_id": sid}
