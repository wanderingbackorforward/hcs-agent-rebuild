"""Tests for MCP Client capability negotiation and feature flags.

Test coverage:
  1. ServerCapabilityProfile parsing from MCP SDK types.
  2. ClientFeatureFlags env var overrides (force-disable).
  3. LocalMCPClient initialize + capability synthesis.
  4. KnowledgeToolBroker capability check — tools_enabled=False raises MCPError.
  5. LocalMCPClient → KnowledgeToolBroker → call_tool full loop (local server).
  6. Capability profile with all features disabled (third-party server scenario).
"""
import asyncio
import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from mcp import types
from mcp_client import (
    MCPClientBase,
    ServerCapabilityProfile,
    ClientFeatureFlags,
    LocalMCPClient,
)
from mcp_server.protocol_handler import ProtocolHandler, _register_default_tools
from mcp_server.errors import MCPError


# ---------------------------------------------------------------------------
# 1. ServerCapabilityProfile parsing
# ---------------------------------------------------------------------------

class TestServerCapabilityProfile:
    def test_from_none_returns_all_false(self):
        profile = ServerCapabilityProfile.from_server_capabilities(None)
        assert not profile.tools_enabled
        assert not profile.resources_enabled
        assert not profile.prompts_enabled
        assert not profile.logging_enabled

    def test_from_full_capabilities(self):
        """Server that advertises all capabilities."""
        caps = types.ServerCapabilities(
            tools=types.ToolsCapability(listChanged=True),
            resources=types.ResourcesCapability(subscribe=True, listChanged=True),
            prompts=types.PromptsCapability(listChanged=True),
            logging=types.LoggingCapability(),
        )
        profile = ServerCapabilityProfile.from_server_capabilities(caps)
        assert profile.tools_enabled
        assert profile.resources_enabled
        assert profile.prompts_enabled
        assert profile.logging_enabled
        assert profile.tool_list_changed
        assert profile.resource_subscriptions
        assert profile.resource_list_changed
        assert profile.prompt_list_changed

    def test_from_tools_only(self):
        """Server that only supports tools (common for third-party servers)."""
        caps = types.ServerCapabilities(
            tools=types.ToolsCapability(),
        )
        profile = ServerCapabilityProfile.from_server_capabilities(caps)
        assert profile.tools_enabled
        assert not profile.resources_enabled
        assert not profile.prompts_enabled
        assert not profile.logging_enabled

    def test_from_dict_local_server(self):
        """Synthesize profile from a dict (local server case)."""
        profile = ServerCapabilityProfile.from_dict({
            "tools": True,
            "resources": False,
            "prompts": False,
            "logging": True,
        })
        assert profile.tools_enabled
        assert not profile.resources_enabled
        assert profile.logging_enabled

    def test_summary_string(self):
        profile = ServerCapabilityProfile(tools_enabled=True, logging_enabled=True)
        s = profile.summary()
        assert "tools" in s
        assert "logging" in s
        assert "resources" not in s


# ---------------------------------------------------------------------------
# 2. ClientFeatureFlags env var overrides
# ---------------------------------------------------------------------------

class TestClientFeatureFlags:
    def test_flags_match_profile(self):
        """Flags should match server profile when no env overrides."""
        profile = ServerCapabilityProfile(
            tools_enabled=True,
            resources_enabled=True,
            prompts_enabled=False,
            logging_enabled=True,
        )
        flags = ClientFeatureFlags.from_profile(profile, env={})
        assert flags.tools_enabled
        assert flags.resources_enabled
        assert not flags.prompts_enabled
        assert flags.logging_enabled

    def test_env_override_disables_tools(self):
        """MCP_DISABLE_TOOLS=true should force-disable tools."""
        profile = ServerCapabilityProfile(tools_enabled=True)
        flags = ClientFeatureFlags.from_profile(
            profile, env={"MCP_DISABLE_TOOLS": "true"}
        )
        assert not flags.tools_enabled

    def test_env_override_disables_logging(self):
        profile = ServerCapabilityProfile(logging_enabled=True)
        flags = ClientFeatureFlags.from_profile(
            profile, env={"MCP_DISABLE_LOGGING": "1"}
        )
        assert not flags.logging_enabled

    def test_env_cannot_force_enable(self):
        """Env vars can only disable, never enable."""
        profile = ServerCapabilityProfile(tools_enabled=False)
        flags = ClientFeatureFlags.from_profile(
            profile, env={"MCP_DISABLE_TOOLS": "false"}
        )
        # Server said no tools → flags must be False regardless of env
        assert not flags.tools_enabled

    def test_multiple_overrides(self):
        profile = ServerCapabilityProfile(
            tools_enabled=True,
            resources_enabled=True,
            prompts_enabled=True,
            logging_enabled=True,
        )
        flags = ClientFeatureFlags.from_profile(
            profile,
            env={
                "MCP_DISABLE_TOOLS": "yes",
                "MCP_DISABLE_LOGGING": "on",
            },
        )
        assert not flags.tools_enabled
        assert flags.resources_enabled  # not overridden
        assert flags.prompts_enabled    # not overridden
        assert not flags.logging_enabled


# ---------------------------------------------------------------------------
# 3. LocalMCPClient initialize + capability synthesis
# ---------------------------------------------------------------------------

class TestLocalMCPClient:
    @pytest.fixture
    def handler_with_tools(self):
        handler = ProtocolHandler(
            server_name="test-server",
            server_version="0.1.0",
        )
        _register_default_tools(handler)
        return handler

    @pytest.mark.asyncio
    async def test_initialize_synthesizes_capabilities(self, handler_with_tools):
        """LocalMCPClient should detect tools/resources/prompts from handler."""
        client = LocalMCPClient(handler_with_tools, server_name="test")
        profile = await client.initialize()

        assert client.initialized
        assert profile.tools_enabled  # default tools are registered
        assert profile.logging_enabled  # our server supports logging

    @pytest.mark.asyncio
    async def test_get_feature_flags_after_init(self, handler_with_tools):
        client = LocalMCPClient(handler_with_tools)
        await client.initialize()
        flags = client.get_feature_flags()
        assert flags.tools_enabled

    @pytest.mark.asyncio
    async def test_get_feature_flags_before_init_returns_permissive_default(self, handler_with_tools):
        """Before initialize(), flags should be permissive (backward compat)."""
        client = LocalMCPClient(handler_with_tools)
        flags = client.get_feature_flags()
        # Permissive default — all True
        assert flags.tools_enabled

    @pytest.mark.asyncio
    async def test_call_tool_through_client(self, handler_with_tools):
        """End-to-end: initialize → call_tool → get result."""
        client = LocalMCPClient(handler_with_tools)
        await client.initialize()

        result = await client.call_tool("query_knowledge_hub", {
            "query": "test query",
            "top_k": 3,
        })
        assert isinstance(result, types.CallToolResult)
        # Tool should return text content
        assert len(result.content) > 0

    @pytest.mark.asyncio
    async def test_list_tools(self, handler_with_tools):
        client = LocalMCPClient(handler_with_tools)
        await client.initialize()
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        assert "list_collections" in tool_names
        assert "query_knowledge_hub" in tool_names
        assert "get_document_summary" in tool_names

    @pytest.mark.asyncio
    async def test_is_tool_available(self, handler_with_tools):
        client = LocalMCPClient(handler_with_tools)
        await client.initialize()
        assert await client.is_tool_available("query_knowledge_hub")
        assert not await client.is_tool_available("nonexistent_tool")

    @pytest.mark.asyncio
    async def test_empty_handler_no_tools(self):
        """A handler with no tools should report tools_enabled=False."""
        handler = ProtocolHandler(server_name="empty", server_version="0.0.1")
        # Don't register any tools
        client = LocalMCPClient(handler)
        profile = await client.initialize()
        assert not profile.tools_enabled


# ---------------------------------------------------------------------------
# 4. KnowledgeToolBroker capability check
# ---------------------------------------------------------------------------

class TestKnowledgeToolBrokerCapabilities:
    @pytest.fixture
    def handler_with_tools(self):
        handler = ProtocolHandler(server_name="test", server_version="0.1.0")
        _register_default_tools(handler)
        return handler

    @pytest.mark.asyncio
    async def test_broker_initializes_client(self, handler_with_tools):
        from agents.knowledge_qa.tool_broker import KnowledgeToolBroker
        broker = KnowledgeToolBroker(
            mcp_client=LocalMCPClient(handler_with_tools)
        )
        flags = await broker.ensure_initialized()
        assert flags.tools_enabled

    @pytest.mark.asyncio
    async def test_broker_raises_on_no_tools(self):
        """Broker should raise MCPError when tools_enabled=False."""
        from agents.knowledge_qa.tool_broker import KnowledgeToolBroker

        # Create a handler with no tools
        handler = ProtocolHandler(server_name="empty", server_version="0.0.1")
        broker = KnowledgeToolBroker(
            mcp_client=LocalMCPClient(handler)
        )

        with pytest.raises(MCPError) as exc_info:
            await broker.query_knowledge_hub(query="test")

        assert exc_info.value.error_type == "capability_not_supported"

    @pytest.mark.asyncio
    async def test_broker_call_tool_success(self, handler_with_tools):
        """Broker should return dict payload when tool succeeds.

        Uses list_collections (no external API dependency) to avoid
        MiniMax embedding errors in CI.
        """
        from agents.knowledge_qa.tool_broker import KnowledgeToolBroker
        broker = KnowledgeToolBroker(
            mcp_client=LocalMCPClient(handler_with_tools)
        )
        result = await broker.list_collections()
        assert isinstance(result, dict)
        assert result.get("_tool_name") == "list_collections"


# ---------------------------------------------------------------------------
# 5. Third-party server scenario simulation
# ---------------------------------------------------------------------------

class TestThirdPartyServerScenario:
    """Simulate a third-party MCP Server that only supports tools, not logging."""

    @pytest.mark.asyncio
    async def test_tools_only_server(self):
        """Server with only tools capability — logging flag should be False."""
        caps = types.ServerCapabilities(
            tools=types.ToolsCapability(),
        )
        profile = ServerCapabilityProfile.from_server_capabilities(caps)
        flags = ClientFeatureFlags.from_profile(profile, env={})

        assert flags.tools_enabled
        assert not flags.logging_enabled
        # Upper layer would skip logging requests — pure switch, no glue code.

    @pytest.mark.asyncio
    async def test_no_capabilities_server(self):
        """Server that supports nothing — all flags should be False."""
        profile = ServerCapabilityProfile.from_server_capabilities(None)
        flags = ClientFeatureFlags.from_profile(profile, env={})

        assert not flags.tools_enabled
        assert not flags.resources_enabled
        assert not flags.prompts_enabled
        assert not flags.logging_enabled

    @pytest.mark.asyncio
    async def test_operator_force_disable_tools(self):
        """Operator can force-disable tools even if server supports them."""
        caps = types.ServerCapabilities(
            tools=types.ToolsCapability(),
            logging=types.LoggingCapability(),
        )
        profile = ServerCapabilityProfile.from_server_capabilities(caps)
        flags = ClientFeatureFlags.from_profile(
            profile,
            env={"MCP_DISABLE_TOOLS": "true"},
        )
        # Server supports tools, but operator disabled them
        assert not flags.tools_enabled
        assert flags.logging_enabled  # logging still works
