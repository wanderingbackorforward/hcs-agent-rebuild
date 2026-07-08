"""MCP Protocol Handler for tool, resource, and prompt registration.

Supports all three MCP primitive types:
  - Tool:     LLM-invoked actions (query/execute/modify)
  - Resource: passively exposed data (LLM reads, does not "call")
  - Prompt:   predefined prompt templates with arguments

Supports RBAC: each tool declares which agent roles are allowed to call it.
At execution time, the caller must pass agent_name; if the agent is not in
the tool's allowed_agents list, the call is denied with a permission error.
"""
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from mcp import types
from mcp.server.lowlevel import Server

from config.audit import audit_event, get_agent_name
from mcp_server.errors import format_error

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable[..., Any]
    # RBAC: which agent roles may call this tool.
    # Empty set = public (any caller allowed). This is the default
    # for backward compat: existing tools remain callable by all.
    allowed_agents: Set[str] = field(default_factory=set)


@dataclass
class ResourceDefinition:
    """A passively exposed data resource that the LLM can read.

    Unlike Tools (actively invoked), Resources are discovered via
    resources/list and read via resources/read. They expose static
    or contextual data — configs, catalogs, status snapshots.
    """
    uri: str
    name: str
    description: str
    handler: Callable[[], str]
    mime_type: str = "text/plain"


@dataclass
class PromptDefinition:
    """A predefined prompt template with named arguments.

    Prompts standardize how the LLM is instructed for recurring tasks
    (e.g. RAG answering, intent classification). The host fills in
    arguments and sends the rendered message to the LLM.
    """
    name: str
    description: str
    arguments: List[types.PromptArgument]
    handler: Callable[..., str]


@dataclass
class ProtocolHandler:
    server_name: str
    server_version: str
    tools: Dict[str, ToolDefinition] = field(default_factory=dict)
    resources: Dict[str, ResourceDefinition] = field(default_factory=dict)
    prompts: Dict[str, PromptDefinition] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Tool registration & execution
    # ------------------------------------------------------------------
    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
        allowed_agents: Optional[Set[str]] = None,
    ) -> None:
        if name in self.tools:
            raise ValueError(f"Tool '{name}' is already registered")
        self.tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            allowed_agents=allowed_agents or set(),
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
        agent_name = get_agent_name()

        if name not in self.tools:
            audit_event(
                layer="tool_server",
                event_type="error",
                message=f"tool not found: {name}",
                data={"tool": name, "agent": agent_name},
                level=40,
            )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: Tool '{name}' not found")],
                isError=True,
            )
        tool = self.tools[name]

        # RBAC check: if the tool has an allowed_agents list, the caller
        # must be in it. Empty set = public (backward compat).
        if tool.allowed_agents and agent_name not in tool.allowed_agents:
            audit_event(
                layer="tool_server",
                event_type="error",
                message=f"permission denied: agent='{agent_name}' tool='{name}'",
                data={"tool": name, "agent": agent_name, "allowed": list(tool.allowed_agents)},
                level=40,
            )
            return types.CallToolResult(
                content=[types.TextContent(
                    type="text",
                    text=f"Error: permission_denied; agent '{agent_name}' is not authorized to call '{name}'",
                )],
                isError=True,
            )

        # Audit: tool call accepted (mask sensitive args)
        from config.audit import mask_sensitive
        audit_event(
            layer="tool_server",
            event_type="tool_call",
            message=f"executing tool: {name}",
            data={"tool": name, "agent": agent_name, "arguments": mask_sensitive(arguments)},
        )

        try:
            result = await tool.handler(**arguments)
            audit_event(
                layer="tool_server",
                event_type="tool_result",
                message=f"tool completed: {name}",
                data={"tool": name, "is_error": False},
            )
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
            audit_event(
                layer="tool_server",
                event_type="error",
                message=f"invalid params for {name}: {e}",
                data={"tool": name},
                level=40,
            )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: Invalid parameters - {e}")],
                isError=True,
            )
        except Exception as e:
            audit_event(
                layer="tool_server",
                event_type="error",
                message=f"internal error in {name}: {type(e).__name__}",
                data={"tool": name},
                level=40,
            )
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=f"Error: Internal server error while executing '{name}'")],
                isError=True,
            )

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "tools": {},
            "resources": {},
            "prompts": {},
        }

    # ------------------------------------------------------------------
    # Resource registration & reading
    # ------------------------------------------------------------------
    def register_resource(
        self,
        uri: str,
        name: str,
        description: str,
        handler: Callable[[], str],
        mime_type: str = "text/plain",
    ) -> None:
        if uri in self.resources:
            raise ValueError(f"Resource '{uri}' is already registered")
        self.resources[uri] = ResourceDefinition(
            uri=uri,
            name=name,
            description=description,
            handler=handler,
            mime_type=mime_type,
        )

    def get_resource_schemas(self) -> List[types.Resource]:
        return [
            types.Resource(
                uri=res.uri,
                name=res.name,
                description=res.description,
                mimeType=res.mime_type,
            )
            for res in self.resources.values()
        ]

    async def read_resource(self, uri: str) -> str:
        agent_name = get_agent_name()
        if uri not in self.resources:
            audit_event(
                layer="tool_server",
                event_type="error",
                message=f"resource not found: {uri}",
                data={"uri": uri, "agent": agent_name},
                level=40,
            )
            raise ValueError(f"Resource '{uri}' not found")

        audit_event(
            layer="tool_server",
            event_type="resource_read",
            message=f"reading resource: {uri}",
            data={"uri": uri, "agent": agent_name},
        )
        try:
            content = self.resources[uri].handler()
            audit_event(
                layer="tool_server",
                event_type="resource_result",
                message=f"resource read completed: {uri}",
                data={"uri": uri, "length": len(content)},
            )
            return content
        except Exception as e:
            err = format_error(e, context=f"read_resource:{uri}")
            audit_event(
                layer="tool_server",
                event_type="error",
                message=f"resource read failed: {uri}",
                data={"uri": uri, "error": err.error_type},
                level=40,
            )
            raise

    # ------------------------------------------------------------------
    # Prompt registration & rendering
    # ------------------------------------------------------------------
    def register_prompt(
        self,
        name: str,
        description: str,
        arguments: List[types.PromptArgument],
        handler: Callable[..., str],
    ) -> None:
        if name in self.prompts:
            raise ValueError(f"Prompt '{name}' is already registered")
        self.prompts[name] = PromptDefinition(
            name=name,
            description=description,
            arguments=arguments,
            handler=handler,
        )

    def get_prompt_schemas(self) -> List[types.Prompt]:
        return [
            types.Prompt(
                name=p.name,
                description=p.description,
                arguments=p.arguments,
            )
            for p in self.prompts.values()
        ]

    async def get_prompt(
        self,
        name: str,
        arguments: Optional[Dict[str, str]] = None,
    ) -> types.GetPromptResult:
        agent_name = get_agent_name()
        if name not in self.prompts:
            audit_event(
                layer="tool_server",
                event_type="error",
                message=f"prompt not found: {name}",
                data={"prompt": name, "agent": agent_name},
                level=40,
            )
            raise ValueError(f"Prompt '{name}' not found")

        prompt = self.prompts[name]
        audit_event(
            layer="tool_server",
            event_type="prompt_get",
            message=f"rendering prompt: {name}",
            data={"prompt": name, "agent": agent_name},
        )
        try:
            rendered = prompt.handler(**(arguments or {}))
            return types.GetPromptResult(
                description=prompt.description,
                messages=[
                    types.PromptMessage(
                        role="user",
                        content=types.TextContent(type="text", text=rendered),
                    )
                ],
            )
        except TypeError as e:
            err = format_error(e, context=f"get_prompt:{name}")
            audit_event(
                layer="tool_server",
                event_type="error",
                message=f"invalid prompt arguments for {name}: {e}",
                data={"prompt": name},
                level=40,
            )
            raise
        except Exception as e:
            err = format_error(e, context=f"get_prompt:{name}")
            audit_event(
                layer="tool_server",
                event_type="error",
                message=f"prompt render failed: {name}",
                data={"prompt": name, "error": err.error_type},
                level=40,
            )
            raise


def _register_default_tools(protocol_handler: ProtocolHandler) -> None:
    from mcp_server.tools.query_knowledge_hub import register_tool as register_query_tool
    from mcp_server.tools.list_collections import register_tool as register_list_tool
    from mcp_server.tools.get_document_summary import register_tool as register_summary_tool
    register_query_tool(protocol_handler)
    register_list_tool(protocol_handler)
    register_summary_tool(protocol_handler)


def _register_default_resources(protocol_handler: ProtocolHandler) -> None:
    from mcp_server.resources.server_status import register_resource as register_status
    from mcp_server.resources.knowledge_catalog import register_resource as register_catalog
    from mcp_server.resources.prompt_catalog import register_resource as register_prompt_cat
    register_status(protocol_handler)
    register_catalog(protocol_handler)
    register_prompt_cat(protocol_handler)


def _register_default_prompts(protocol_handler: ProtocolHandler) -> None:
    from mcp_server.prompts.rag_answer import register_prompt as register_rag
    from mcp_server.prompts.classify_intent import register_prompt as register_classify
    register_rag(protocol_handler)
    register_classify(protocol_handler)


def create_mcp_server(
    server_name: str,
    server_version: str,
    protocol_handler: Optional[ProtocolHandler] = None,
    register_tools: bool = True,
    register_resources: bool = True,
    register_prompts: bool = True,
) -> Server:
    if protocol_handler is None:
        protocol_handler = ProtocolHandler(
            server_name=server_name,
            server_version=server_version,
        )
    if register_tools:
        _register_default_tools(protocol_handler)
    if register_resources:
        _register_default_resources(protocol_handler)
    if register_prompts:
        _register_default_prompts(protocol_handler)

    server = Server(server_name)

    @server.list_tools()
    async def handle_list_tools() -> List[types.Tool]:
        return protocol_handler.get_tool_schemas()

    @server.call_tool()
    async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> types.CallToolResult:
        return await protocol_handler.execute_tool(name, arguments)

    @server.list_resources()
    async def handle_list_resources() -> List[types.Resource]:
        return protocol_handler.get_resource_schemas()

    @server.read_resource()
    async def handle_read_resource(uri):
        return await protocol_handler.read_resource(str(uri))

    @server.list_prompts()
    async def handle_list_prompts() -> List[types.Prompt]:
        return protocol_handler.get_prompt_schemas()

    @server.get_prompt()
    async def handle_get_prompt(
        name: str, arguments: Optional[Dict[str, str]] = None
    ) -> types.GetPromptResult:
        return await protocol_handler.get_prompt(name, arguments)

    server._protocol_handler = protocol_handler
    return server
