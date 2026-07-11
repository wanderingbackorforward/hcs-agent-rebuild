"""LocalMCPClient — wraps an in-process ProtocolHandler as an MCP Client.

This is the "local server" case: our own MCP Server runs in the same process,
registered via ProtocolHandler. We know exactly what it supports because we
registered the tools/resources/prompts ourselves.

Capability synthesis:
    - tools_enabled = len(protocol_handler.tools) > 0
    - resources_enabled = len(protocol_handler.resources) > 0
    - prompts_enabled = len(protocol_handler.prompts) > 0
    - logging_enabled = True (our server supports logging notifications)

No network round-trip needed; initialize() is synchronous-equivalent.
"""
import logging
from typing import Any, Dict, List, Optional

from mcp import types

from .base import MCPClientBase
from .capabilities import ServerCapabilityProfile

logger = logging.getLogger(__name__)


class LocalMCPClient(MCPClientBase):
    """MCP Client for an in-process ProtocolHandler (our own server).

    This replaces the old pattern where KnowledgeToolBroker directly created
    a ProtocolHandler and called execute_tool(). Now the Broker goes through
    this Client, which provides capability negotiation and feature flags.
    """

    def __init__(
        self,
        protocol_handler: Any,
        server_name: str = "",
        server_version: str = "",
    ):
        super().__init__(server_name=server_name, server_version=server_version)
        self._handler = protocol_handler

    async def initialize(self) -> ServerCapabilityProfile:
        """Synthesize capabilities from registered primitives.

        For a local server, we don't need a network handshake — we just
        inspect what's registered on the ProtocolHandler.
        """
        has_tools = bool(getattr(self._handler, "tools", {}))
        has_resources = bool(getattr(self._handler, "resources", {}))
        has_prompts = bool(getattr(self._handler, "prompts", {}))

        profile = ServerCapabilityProfile.from_dict({
            "tools": has_tools,
            "resources": has_resources,
            "prompts": has_prompts,
            "logging": True,  # Our server supports logging notifications
            "completions": False,
            "tool_list_changed": False,  # Not implemented yet
            "resource_list_changed": False,
            "prompt_list_changed": False,
        })

        self._set_profile_and_flags(profile)
        return profile

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> types.CallToolResult:
        """Delegate to ProtocolHandler.execute_tool()."""
        return await self._handler.execute_tool(name, arguments)

    async def list_tools(self) -> List[types.Tool]:
        """Return tool schemas from ProtocolHandler."""
        return self._handler.get_tool_schemas()

    async def list_resources(self) -> List[types.Resource]:
        """Return resource schemas from ProtocolHandler."""
        return self._handler.get_resource_schemas()

    async def read_resource(self, uri: str) -> str:
        """Read a resource via ProtocolHandler."""
        return await self._handler.read_resource(uri)

    async def list_prompts(self) -> List[types.Prompt]:
        """Return prompt schemas from ProtocolHandler."""
        return self._handler.get_prompt_schemas()

    async def get_prompt(
        self, name: str, arguments: Optional[Dict[str, str]] = None
    ) -> types.GetPromptResult:
        """Render a prompt via ProtocolHandler."""
        return await self._handler.get_prompt(name, arguments)
