"""RemoteMCPClient — connects to a third-party MCP Server via stdio or SSE.

This is the "third-party Server you don't control" case. The Client performs
a real MCP initialize handshake, receives ServerCapabilities, and caches them
as a ServerCapabilityProfile. Feature flags are then derived from the profile.

Transport selection:
    - stdio: spawn a local process (command + args) and communicate via stdin/stdout.
    - sse:   connect to a remote HTTP+SSE endpoint.

Key design: all capability negotiation happens here. The upper layer
(KnowledgeToolBroker) never talks to the Server directly — it only checks
feature flags and calls call_tool/list_tools through this Client.

Interview talking point: "When connecting to a third-party MCP Server, my
Client does a real initialize handshake. If the Server doesn't advertise
logging or resources, I turn off those feature flags and the upper layer
never sends those requests. Pure client-side switch, no glue code."
"""
import logging
from typing import Any, Dict, List, Optional

from mcp import types

from .base import MCPClientBase
from .capabilities import ServerCapabilityProfile

logger = logging.getLogger(__name__)


class RemoteMCPClient(MCPClientBase):
    """MCP Client for a remote or subprocess MCP Server.

    Uses the MCP SDK's ClientSession for real protocol-level communication,
    including the initialize handshake and capability negotiation.

    Args:
        transport: "stdio" or "sse".
        server_name: Display name for this server (for logging).
        server_version: Expected server version (informational).

    For stdio:
        command: Executable to spawn (e.g., "python" or "node").
        args: List of command-line arguments.
        env: Optional environment variables for the subprocess.

    For SSE:
        url: The SSE endpoint URL (e.g., "http://localhost:8080/sse").
        headers: Optional HTTP headers.
        timeout: HTTP timeout in seconds.
        sse_read_timeout: SSE read timeout in seconds.
    """

    def __init__(
        self,
        transport: str = "stdio",
        server_name: str = "",
        server_version: str = "",
        # stdio params
        command: Optional[str] = None,
        args: Optional[List[str]] = None,
        env: Optional[Dict[str, str]] = None,
        # SSE params
        url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 5.0,
        sse_read_timeout: float = 300.0,
    ):
        super().__init__(server_name=server_name, server_version=server_version)
        self._transport = transport.lower()
        self._command = command
        self._args = args or []
        self._env = env
        self._url = url
        self._headers = headers
        self._timeout = timeout
        self._sse_read_timeout = sse_read_timeout

        # Session state (set during initialize).
        self._session = None
        self._cm_stack = None  # AsyncExitStack for the transport context manager.

    async def initialize(self) -> ServerCapabilityProfile:
        """Perform real MCP initialize handshake with the remote Server.

        This is where capability negotiation happens. The Server returns
        its ServerCapabilities in the InitializeResult, and we parse it
        into a ServerCapabilityProfile.

        Raises:
            ConnectionError: If the transport fails to connect.
            Exception: If the Server returns an error during initialize.
        """
        try:
            if self._transport == "stdio":
                await self._init_stdio()
            elif self._transport == "sse":
                await self._init_sse()
            else:
                raise ValueError(f"Unknown transport: {self._transport}")
        except Exception as e:
            logger.error("RemoteMCPClient initialize failed (transport=%s): %s",
                         self._transport, e)
            raise

        # ClientSession.initialize() returns InitializeResult with .capabilities
        result = await self._session.initialize()
        caps = getattr(result, "capabilities", None)
        server_info = getattr(result, "serverInfo", None)

        if server_info:
            self._server_name = getattr(server_info, "name", self._server_name)
            self._server_version = getattr(server_info, "version", self._server_version)

        profile = ServerCapabilityProfile.from_server_capabilities(caps)
        self._set_profile_and_flags(profile)

        logger.info(
            "RemoteMCPClient connected: server='%s' version='%s' transport='%s' capabilities=[%s]",
            self._server_name, self._server_version, self._transport, profile.summary(),
        )
        return profile

    async def _init_stdio(self) -> None:
        """Establish stdio transport connection to a subprocess MCP Server."""
        from contextlib import AsyncExitStack
        from mcp.client.stdio import stdio_client

        if not self._command:
            raise ValueError("stdio transport requires 'command' parameter")

        self._cm_stack = AsyncExitStack()
        await self._cm_stack.__aenter__()

        read_stream, write_stream = await self._cm_stack.enter_async_context(
            stdio_client(
                command=self._command,
                args=self._args,
                env=self._env,
            )
        )

        from mcp.client.session import ClientSession
        self._session = ClientSession(read_stream, write_stream)

    async def _init_sse(self) -> None:
        """Establish SSE transport connection to a remote MCP Server."""
        from contextlib import AsyncExitStack
        from mcp.client.sse import sse_client

        if not self._url:
            raise ValueError("sse transport requires 'url' parameter")

        self._cm_stack = AsyncExitStack()
        await self._cm_stack.__aenter__()

        read_stream, write_stream = await self._cm_stack.enter_async_context(
            sse_client(
                url=self._url,
                headers=self._headers,
                timeout=self._timeout,
                sse_read_timeout=self._sse_read_timeout,
            )
        )

        from mcp.client.session import ClientSession
        self._session = ClientSession(read_stream, write_stream)

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> types.CallToolResult:
        """Call a tool on the remote Server via ClientSession."""
        if self._session is None:
            raise RuntimeError("Client not initialized; call initialize() first")
        return await self._session.call_tool(name, arguments)

    async def list_tools(self) -> List[types.Tool]:
        """List tools from the remote Server."""
        if self._session is None:
            raise RuntimeError("Client not initialized; call initialize() first")
        result = await self._session.list_tools()
        return result.tools

    async def list_resources(self) -> List[types.Resource]:
        """List resources from the remote Server."""
        if self._session is None:
            raise RuntimeError("Client not initialized; call initialize() first")
        result = await self._session.list_resources()
        return result.resources

    async def read_resource(self, uri: str) -> str:
        """Read a resource from the remote Server."""
        if self._session is None:
            raise RuntimeError("Client not initialized; call initialize() first")
        result = await self._session.read_resource(uri)
        # ReadResourceResult has .contents which is a list of TextContent | BlobContent
        if result.contents:
            first = result.contents[0]
            if hasattr(first, "text"):
                return first.text
        return ""

    async def list_prompts(self) -> List[types.Prompt]:
        """List prompts from the remote Server."""
        if self._session is None:
            raise RuntimeError("Client not initialized; call initialize() first")
        result = await self._session.list_prompts()
        return result.prompts

    async def get_prompt(
        self, name: str, arguments: Optional[Dict[str, str]] = None
    ) -> types.GetPromptResult:
        """Get a rendered prompt from the remote Server."""
        if self._session is None:
            raise RuntimeError("Client not initialized; call initialize() first")
        return await self._session.get_prompt(name, arguments)

    async def close(self) -> None:
        """Close the transport connection and release resources."""
        if self._cm_stack is not None:
            await self._cm_stack.__aexit__(None, None, None)
            self._cm_stack = None
        self._session = None
        self._initialized = False
        logger.info("RemoteMCPClient disconnected: server='%s'", self._server_name)

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
