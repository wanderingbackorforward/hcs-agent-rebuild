"""MCP Protocol Handler for tool registration and execution."""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from mcp import types
from mcp.server.lowlevel import Server


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any]


@dataclass
class ProtocolHandler:
    server_name: str
    server_version: str
    tools: Dict[str, ToolDefinition] = field(default_factory=dict)

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        if name in self.tools:
            raise ValueError(f"Tool '{name}' is already registered")
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
        )

    def get_tool_schemas(self) -> List[types.Tool]:
        return [
            types.Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.input_schema,
            )
            for tool in self.tools.values()
        ]

    async def execute_tool(self, name: str, arguments: Dict[str, Any]) -> types.CallToolResult:
        if name not in self.tools:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: Tool '{name}' not found")],
                isError=True,
            )
        tool = self.tools[name]
        try:
            result = await tool.handler(**arguments)
            if isinstance(result, types.CallToolResult):
                return result
            if isinstance(result, str):
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text=result)],
                    isError=False,
                )
            if isinstance(result, list):
                return types.CallToolResult(content=result, isError=False)
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(result))],
                isError=False,
            )
        except TypeError as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: Invalid parameters - {e}")],
                isError=True,
            )
        except Exception as e:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: Internal server error while executing '{name}'")],
                isError=True,
            )

    def get_capabilities(self) -> Dict[str, Any]:
        return {"tools": {}}


def _register_default_tools(protocol_handler: ProtocolHandler) -> None:
    from mcp_server.tools.query_knowledge_hub import register_tool as register_query_tool
    from mcp_server.tools.list_collections import register_tool as register_list_tool
    from mcp_server.tools.get_document_summary import register_tool as register_summary_tool
    register_query_tool(protocol_handler)
    register_list_tool(protocol_handler)
    register_summary_tool(protocol_handler)


def create_mcp_server(
    server_name: str,
    server_version: str,
    protocol_handler: Optional[ProtocolHandler] = None,
    register_tools: bool = True,
) -> Server:
    if protocol_handler is None:
        protocol_handler = ProtocolHandler(
            server_name=server_name,
            server_version=server_version,
        )
    if register_tools:
        _register_default_tools(protocol_handler)

    server = Server(server_name)

    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        return protocol_handler.get_tool_schemas()

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> types.CallToolResult:
        return await protocol_handler.execute_tool(name, arguments)

    server._protocol_handler = protocol_handler
    return server
