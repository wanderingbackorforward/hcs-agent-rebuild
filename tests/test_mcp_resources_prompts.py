"""MCP Resource & Prompt handler unit tests.

Tests cover:
  - Resource registration, listing, and reading (success + not-found)
  - Prompt registration, listing, and rendering (success + not-found + bad args)
  - ProtocolHandler isolation (separate instances don't leak state)
  - create_mcp_server wiring (all three primitive types registered)

These tests use direct in-process invocation — no MCP stdio transport.
"""
import asyncio
import json

import pytest

from mcp import types
from mcp_server.protocol_handler import (
    ProtocolHandler,
    create_mcp_server,
    ResourceDefinition,
    PromptDefinition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def handler():
    """A fresh ProtocolHandler with no registered primitives."""
    return ProtocolHandler(
        server_name="test-server",
        server_version="0.0.1",
    )


@pytest.fixture
def populated_handler():
    """A ProtocolHandler with default resources and prompts registered."""
    from mcp_server.resources.server_status import register_resource as reg_status
    from mcp_server.resources.prompt_catalog import register_resource as reg_prompt_cat
    from mcp_server.prompts.rag_answer import register_prompt as reg_rag
    from mcp_server.prompts.classify_intent import register_prompt as reg_classify

    h = ProtocolHandler(
        server_name="test-server",
        server_version="0.0.1",
    )
    reg_status(h)
    reg_prompt_cat(h)
    reg_rag(h)
    reg_classify(h)
    return h


# ---------------------------------------------------------------------------
# Resource tests
# ---------------------------------------------------------------------------

class TestResourceRegistration:
    def test_register_resource_appears_in_schemas(self, handler):
        handler.register_resource(
            uri="test://foo",
            name="foo",
            description="A test resource.",
            handler=lambda: "hello",
        )
        schemas = handler.get_resource_schemas()
        assert len(schemas) == 1
        assert str(schemas[0].uri) == "test://foo"
        assert schemas[0].name == "foo"
        assert schemas[0].description == "A test resource."

    def test_register_duplicate_uri_raises(self, handler):
        handler.register_resource(
            uri="test://dup",
            name="dup",
            description="dup",
            handler=lambda: "x",
        )
        with pytest.raises(ValueError, match="already registered"):
            handler.register_resource(
                uri="test://dup",
                name="dup2",
                description="dup2",
                handler=lambda: "y",
            )


class TestResourceRead:
    def test_read_resource_returns_content(self, handler):
        handler.register_resource(
            uri="test://data",
            name="data",
            description="data",
            handler=lambda: "resource body",
        )
        content = asyncio.run(handler.read_resource("test://data"))
        assert content == "resource body"

    def test_read_resource_not_found_raises(self, handler):
        with pytest.raises(ValueError, match="not found"):
            asyncio.run(handler.read_resource("test://missing"))

    def test_server_status_resource_returns_json(self, populated_handler):
        content = asyncio.run(
            populated_handler.read_resource("hcs://server/status")
        )
        data = json.loads(content)
        assert data["server_name"] == "test-server"
        assert data["resource_count"] == 2
        assert data["prompt_count"] == 2

    def test_prompt_catalog_resource_lists_templates(self, populated_handler):
        content = asyncio.run(
            populated_handler.read_resource("hcs://prompts/catalog")
        )
        assert "Prompt Catalog" in content
        assert "rag_answer_v1.txt" in content
        assert "classification_v1.txt" in content


# ---------------------------------------------------------------------------
# Prompt tests
# ---------------------------------------------------------------------------

class TestPromptRegistration:
    def test_register_prompt_appears_in_schemas(self, handler):
        handler.register_prompt(
            name="test-prompt",
            description="A test prompt.",
            arguments=[
                types.PromptArgument(name="x", description="arg x", required=True),
            ],
            handler=lambda x: f"result: {x}",
        )
        schemas = handler.get_prompt_schemas()
        assert len(schemas) == 1
        assert schemas[0].name == "test-prompt"
        assert schemas[0].arguments[0].name == "x"

    def test_register_duplicate_name_raises(self, handler):
        handler.register_prompt(
            name="dup",
            description="d",
            arguments=[],
            handler=lambda: "x",
        )
        with pytest.raises(ValueError, match="already registered"):
            handler.register_prompt(
                name="dup",
                description="d2",
                arguments=[],
                handler=lambda: "y",
            )


class TestPromptRender:
    def test_get_prompt_renders_message(self, handler):
        handler.register_prompt(
            name="echo",
            description="Echo prompt.",
            arguments=[
                types.PromptArgument(name="msg", required=True),
            ],
            handler=lambda msg: f"Echo: {msg}",
        )
        result = asyncio.run(handler.get_prompt("echo", {"msg": "hi"}))
        assert isinstance(result, types.GetPromptResult)
        assert result.description == "Echo prompt."
        assert len(result.messages) == 1
        assert result.messages[0].role == "user"
        assert result.messages[0].content.text == "Echo: hi"

    def test_get_prompt_not_found_raises(self, handler):
        with pytest.raises(ValueError, match="not found"):
            asyncio.run(handler.get_prompt("nonexistent"))

    def test_get_prompt_missing_required_arg_raises(self, handler):
        handler.register_prompt(
            name="needs-arg",
            description="d",
            arguments=[
                types.PromptArgument(name="val", required=True),
            ],
            handler=lambda val: f"got {val}",
        )
        # Missing required arg now raises a sanitized ValueError (not raw
        # TypeError) to prevent leaking internal exception details to stdout.
        with pytest.raises(ValueError, match="Failed to render prompt 'needs-arg'"):
            asyncio.run(handler.get_prompt("needs-arg", {}))

    def test_rag_answer_prompt_renders_template(self, populated_handler):
        result = asyncio.run(
            populated_handler.get_prompt(
                "rag-answer",
                {"context": "CTX", "query": "Q?"},
            )
        )
        text = result.messages[0].content.text
        assert "CTX" in text
        assert "Q?" in text

    def test_classify_intent_prompt_optional_arg(self, populated_handler):
        # Without optional history_text — should still work.
        result = asyncio.run(
            populated_handler.get_prompt(
                "classify-intent",
                {"user_input": "查一下华东区环境"},
            )
        )
        text = result.messages[0].content.text
        assert "查一下华东区环境" in text

        # With optional history_text.
        result2 = asyncio.run(
            populated_handler.get_prompt(
                "classify-intent",
                {"user_input": "查环境", "history_text": "历史对话"},
            )
        )
        text2 = result2.messages[0].content.text
        assert "历史对话" in text2


# ---------------------------------------------------------------------------
# create_mcp_server integration
# ---------------------------------------------------------------------------

class TestCreateMCPServer:
    def test_all_three_primitives_registered(self):
        server = create_mcp_server("test", "0.1.0")
        ph = server._protocol_handler
        # 3 default tools
        assert len(ph.tools) == 3
        assert "query_knowledge_hub" in ph.tools
        # 3 default resources
        assert len(ph.resources) == 3
        assert "hcs://server/status" in ph.resources
        # 2 default prompts
        assert len(ph.prompts) == 2
        assert "rag-answer" in ph.prompts

    def test_disable_resources_and_prompts(self):
        server = create_mcp_server(
            "test", "0.1.0",
            register_resources=False,
            register_prompts=False,
        )
        ph = server._protocol_handler
        assert len(ph.tools) == 3
        assert len(ph.resources) == 0
        assert len(ph.prompts) == 0
