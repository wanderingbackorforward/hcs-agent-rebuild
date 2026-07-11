"""MCP Client abstract base class.

Defines the unified interface that both LocalMCPClient and RemoteMCPClient
implement. The key contract:

1. ``initialize()`` — performs capability negotiation, caches the profile,
   and derives ClientFeatureFlags. Must be called before any tool/resource/prompt call.
2. ``call_tool(name, args)`` — executes a tool on the Server.
3. ``list_tools()`` — returns available tool schemas.
4. ``get_feature_flags()`` — returns the cached ClientFeatureFlags.
5. ``is_tool_available(name)`` — checks if a specific tool exists on the Server.

Upper layers (KnowledgeToolBroker) check feature flags BEFORE making requests.
If tools_enabled is False, the Broker skips the tool path entirely and the
Agent falls back to the legacy retriever. No glue code, no simulation.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from mcp import types

from .capabilities import ServerCapabilityProfile, ClientFeatureFlags

logger = logging.getLogger(__name__)


class MCPClientBase(ABC):
    """Abstract MCP Client — owns capability negotiation and tool execution.

    Lifecycle:
        client = LocalMCPClient(protocol_handler)  # or RemoteMCPClient(...)
        await client.initialize()                   # capability handshake
        flags = client.get_feature_flags()          # query switches
        if flags.tools_enabled:
            result = await client.call_tool("query_knowledge_hub", {...})
    """

    def __init__(self, server_name: str = "", server_version: str = ""):
        self._server_name = server_name
        self._server_version = server_version
        self._profile: Optional[ServerCapabilityProfile] = None
        self._flags: Optional[ClientFeatureFlags] = None
        self._initialized = False
        self._available_tools: Optional[List[str]] = None

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def server_name(self) -> str:
        return self._server_name

    @abstractmethod
    async def initialize(self) -> ServerCapabilityProfile:
        """Perform capability negotiation with the Server.

        For LocalMCPClient: synthesizes capabilities from registered primitives.
        For RemoteMCPClient: calls the MCP SDK's ClientSession.initialize().

        After this call, ``get_feature_flags()`` and ``is_tool_available()``
        are safe to query.

        Returns:
            The ServerCapabilityProfile negotiated from the Server.
        """
        ...

    @abstractmethod
    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> types.CallToolResult:
        """Execute a tool on the Server.

        Args:
            name: Tool name (must be in the Server's tool list).
            arguments: Tool input arguments matching the tool's input schema.

        Returns:
            CallToolResult from the Server. Caller is responsible for checking
            ``result.isError``.
        """
        ...

    @abstractmethod
    async def list_tools(self) -> List[types.Tool]:
        """Return the list of tools available on the Server.

        Returns an empty list if the Server doesn't support tools.
        """
        ...

    @abstractmethod
    async def list_resources(self) -> List[types.Resource]:
        """Return the list of resources available on the Server.

        Returns an empty list if the Server doesn't support resources.
        """
        ...

    @abstractmethod
    async def read_resource(self, uri: str) -> str:
        """Read a resource by URI from the Server.

        Raises if the Server doesn't support resources or the URI is unknown.
        """
        ...

    @abstractmethod
    async def list_prompts(self) -> List[types.Prompt]:
        """Return the list of prompts available on the Server.

        Returns an empty list if the Server doesn't support prompts.
        """
        ...

    @abstractmethod
    async def get_prompt(
        self, name: str, arguments: Optional[Dict[str, str]] = None
    ) -> types.GetPromptResult:
        """Render a prompt template from the Server.

        Raises if the Server doesn't support prompts or the name is unknown.
        """
        ...

    # ------------------------------------------------------------------
    # Convenience methods (not abstract — implemented on top of the above)
    # ------------------------------------------------------------------

    def get_feature_flags(self) -> ClientFeatureFlags:
        """Return cached feature flags. Must be called after initialize().

        If called before initialize(), returns a permissive default where
        all capabilities are enabled (backward-compatible with old code
        that didn't do capability negotiation).
        """
        if self._flags is None:
            logger.warning("get_feature_flags() called before initialize(); using permissive default")
            return ClientFeatureFlags()
        return self._flags

    def get_capability_profile(self) -> Optional[ServerCapabilityProfile]:
        """Return the cached ServerCapabilityProfile, or None if not initialized."""
        return self._profile

    async def is_tool_available(self, name: str) -> bool:
        """Check if a specific tool exists on the Server.

        Caches the tool list after the first call. Returns False if the
        Server doesn't support tools or the tool name is not in the list.
        """
        if not self._flags or not self._flags.tools_enabled:
            return False
        if self._available_tools is None:
            tools = await self.list_tools()
            self._available_tools = [t.name for t in tools]
        return name in self._available_tools

    def _set_profile_and_flags(self, profile: ServerCapabilityProfile) -> None:
        """Internal: cache the profile and derive feature flags."""
        self._profile = profile
        self._flags = ClientFeatureFlags.from_profile(profile)
        self._initialized = True
        logger.info(
            "MCP Client initialized: server='%s' capabilities=[%s] flags=%s",
            self._server_name,
            profile.summary(),
            self._flags.to_dict(),
        )
