"""Tests for SSE event protocol and stream formatting."""
import asyncio
import json
import pytest

from config.sse_protocol import SSEEvent, collect_text, format_sse_stream


class TestSSEEvent:
    def test_status_factory(self):
        e = SSEEvent.status("classifying", "正在理解需求...")
        assert e.type == "status"
        assert e.data["stage"] == "classifying"
        assert e.data["message"] == "正在理解需求..."
        assert "timestamp" in e.data

    def test_token_factory(self):
        e = SSEEvent.token("hello")
        assert e.type == "token"
        assert e.data["content"] == "hello"
        assert e.text == "hello"

    def test_decision_factory(self):
        e = SSEEvent.decision(intent_type="knowledge_qa", confidence=0.95)
        assert e.type == "decision"
        assert e.data["intent_type"] == "knowledge_qa"
        assert e.data["confidence"] == 0.95

    def test_error_factory(self):
        e = SSEEvent.error("TimeoutError", "超时", retryable=True, suggestion="重试")
        assert e.type == "error"
        assert e.data["error_type"] == "TimeoutError"
        assert e.data["retryable"] is True

    def test_done_factory(self):
        e = SSEEvent.done(session_id="abc123")
        assert e.type == "done"
        assert e.data["session_id"] == "abc123"

    def test_text_property_non_token(self):
        """Non-token events return empty string for .text."""
        assert SSEEvent.status("s", "m").text == ""
        assert SSEEvent.error("E", "m").text == ""
        assert SSEEvent.done().text == ""

    def test_to_sse_format(self):
        e = SSEEvent.token("hi")
        e.seq = 5
        sse = e.to_sse()
        assert sse.startswith("event: token\n")
        assert "data: " in sse
        assert sse.endswith("\n\n")
        payload = json.loads(sse.split("data: ")[1].strip())
        assert payload["content"] == "hi"

    def test_to_sse_chinese(self):
        """Chinese characters should not be escaped (ensure_ascii=False)."""
        e = SSEEvent.status("classifying", "正在理解需求")
        sse = e.to_sse()
        assert "正在理解需求" in sse

    def test_str_fallback(self):
        """__str__ returns token text for backward compat."""
        assert str(SSEEvent.token("hello")) == "hello"
        assert str(SSEEvent.status("s", "m")) == ""


class TestCollectText:
    @pytest.mark.asyncio
    async def test_collect_plain_strings(self):
        async def gen():
            yield "hello "
            yield "world"
        assert await collect_text(gen()) == "hello world"

    @pytest.mark.asyncio
    async def test_collect_mixed_sse_and_str(self):
        async def gen():
            yield SSEEvent.status("s", "processing")
            yield SSEEvent.token("hello ")
            yield "world"
            yield SSEEvent.done()
        assert await collect_text(gen()) == "hello world"

    @pytest.mark.asyncio
    async def test_collect_empty(self):
        async def gen():
            return
            yield  # make it an async generator
        assert await collect_text(gen()) == ""


class TestFormatSSEStream:
    @pytest.mark.asyncio
    async def test_format_assigns_seq(self):
        async def gen():
            yield "a"
            yield SSEEvent.token("b")
            yield "c"

        chunks = []
        async for chunk in format_sse_stream(gen(), session_id=""):
            chunks.append(chunk)

        # Each chunk should contain an event with increasing seq.
        seqs = []
        for chunk in chunks:
            lines = chunk.strip().split("\n")
            data_line = [l for l in lines if l.startswith("data: ")][0]
            data = json.loads(data_line[6:])
            seqs.append(data.get("content", ""))

        assert seqs == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_format_wraps_strings_as_token(self):
        async def gen():
            yield "hello"
        chunks = []
        async for chunk in format_sse_stream(gen(), session_id=""):
            chunks.append(chunk)
        assert "event: token" in chunks[0]
        assert "hello" in chunks[0]

    @pytest.mark.asyncio
    async def test_format_preserves_sse_events(self):
        async def gen():
            yield SSEEvent.status("classifying", "thinking")
        chunks = []
        async for chunk in format_sse_stream(gen(), session_id=""):
            chunks.append(chunk)
        assert "event: status" in chunks[0]
        assert "thinking" in chunks[0]
