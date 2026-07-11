"""Knowledge MCP tool broker for KnowledgeQAAgent.

Uses ProtocolHandler.execute_tool() as the unified MCP execution entry,
rather than importing tool handlers directly.
"""
import json
from typing import Any, Dict, Optional

from config.audit import set_trace_context
from config.settings import app_settings
from mcp_server.errors import MCPError
from mcp_server.protocol_handler import ProtocolHandler, _register_default_tools


class KnowledgeToolBroker:
    """Thin MCP client facade for the three knowledge tools."""

    def __init__(self, protocol_handler: Optional[ProtocolHandler] = None):
        if protocol_handler is None:
            protocol_handler = ProtocolHandler(
                server_name=app_settings.mcp_server_name,
                server_version=app_settings.app_version,
            )
            _register_default_tools(protocol_handler)
        self.protocol_handler = protocol_handler

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        set_trace_context(agent_name="knowledge_qa")
        result = await self.protocol_handler.execute_tool(name, arguments)
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
