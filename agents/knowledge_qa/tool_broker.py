"""Knowledge MCP tool broker for KnowledgeQAAgent.

Previously this module directly created a ProtocolHandler and called
execute_tool(). Now it goes through MCPClientBase, which provides:
  - Capability negotiation (initialize handshake)
  - ClientFeatureFlags (pure client-side switches)
  - Support for both local (in-process) and remote (third-party) MCP Servers

If the connected Server doesn't support tools (tools_enabled=False), the
Broker raises MCPError and the Agent falls back to the legacy retriever.
No glue code, no simulation — just a boolean check.
"""
import json
import logging
from typing import Any, Dict, Optional

from config.audit import set_trace_context
from config.settings import app_settings
from mcp_server.errors import MCPError
from mcp_server.protocol_handler import ProtocolHandler, _register_default_tools
from mcp_client import MCPClientBase, LocalMCPClient
from mcp_client.capabilities import ClientFeatureFlags

logger = logging.getLogger(__name__)


class KnowledgeToolBroker:
    """Thin MCP client facade for the three knowledge tools.

    Goes through MCPClientBase instead of directly calling ProtocolHandler.
    This adds capability negotiation and feature flag checks.

    Args:
        mcp_client: An initialized MCPClientBase instance. If None, a
            LocalMCPClient wrapping a default ProtocolHandler is created.
        protocol_handler: Deprecated, kept for backward compat. If
            mcp_client is None and protocol_handler is provided, wraps it
            in a LocalMCPClient.
    """

    def __init__(
        self,
        mcp_client: Optional[MCPClientBase] = None,
        protocol_handler: Optional[ProtocolHandler] = None,
    ):
        if mcp_client is not None:
            self._client = mcp_client
        else:
            # Backward-compatible: create a LocalMCPClient with default tools.
            if protocol_handler is None:
                protocol_handler = ProtocolHandler(
                    server_name=app_settings.mcp_server_name,
                    server_version=app_settings.app_version,
                )
                _register_default_tools(protocol_handler)
            self._client = LocalMCPClient(
                protocol_handler=protocol_handler,
                server_name=app_settings.mcp_server_name,
                server_version=app_settings.app_version,
            )

        self._initialized = False

    @property
    def mcp_client(self) -> MCPClientBase:
        """Direct access to the underlying MCP Client (for capability queries)."""
        return self._client

    async def ensure_initialized(self) -> ClientFeatureFlags:
        """Initialize the MCP Client if not yet done, return feature flags.

        This is where capability negotiation happens. After this call,
        get_feature_flags() reflects what the Server actually supports.
        """
        if not self._initialized:
            await self._client.initialize()
            self._initialized = True
        return self._client.get_feature_flags()

    def get_feature_flags(self) -> ClientFeatureFlags:
        """Return cached feature flags (permissive default if not initialized)."""
        return self._client.get_feature_flags()

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool through the MCP Client.

        Raises MCPError if:
          - tools_enabled is False (Server doesn't support tools)
          - the tool execution returns isError=True
        """
        set_trace_context(agent_name="knowledge_qa")
        flags = await self.ensure_initialized()

        # Pure client-side switch: if Server doesn't support tools, bail out.
        # The upper layer (KnowledgeQAAgent) catches this and falls back.
        if not flags.tools_enabled:
            raise MCPError(
                error_type="capability_not_supported",
                message="MCP Server does not support tools capability. "
                        "Feature flag tools_enabled=False.",
                details={"tool": name, "flags": flags.to_dict()},
            )

        result = await self._client.call_tool(name, arguments)
        text = "\n".join(
            getattr(item, "text", "")
            for item in getattr(result, "content", [])
            if getattr(item, "type", "") == "text"
        ).strip()
        if result.isError:
            raise MCPError(
                error_type="tool_execution_failed",
                message=f"Tool '{name}' execution failed.",
                details={"tool": name, "payload": text},
            )
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {"raw_text": text}
        if not isinstance(payload, dict):
            payload = {"value": payload}
        payload["_tool_name"] = name
        return payload

    async def list_collections(self, **kwargs) -> Dict[str, Any]:
        return await self.call_tool("list_collections", kwargs)

    async def query_knowledge_hub(self, **kwargs) -> Dict[str, Any]:
        return await self.call_tool("query_knowledge_hub", kwargs)

    async def get_document_summary(self, **kwargs) -> Dict[str, Any]:
        return await self.call_tool("get_document_summary", kwargs)
