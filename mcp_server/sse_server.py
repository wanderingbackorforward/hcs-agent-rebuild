"""MCP Server with SSE (Server-Sent Events) transport.

Supports both stdio (local CLI) and SSE (remote/browser) transport modes.
Switch via MCP_TRANSPORT environment variable.

Interview talking point: "My MCP Server supports two transport modes —
stdio for local CLI usage, SSE for remote access and browser connections.
SSE enables cross-network access and multi-client scenarios. Switch via
environment variable, no code changes needed."
"""
import asyncio
import logging
import os
import sys

from mcp.server.sse import SseServerTransport
from config.settings import app_settings
from mcp_server.protocol_handler import create_mcp_server

logger = logging.getLogger(__name__)

SERVER_NAME = app_settings.mcp_server_name
SERVER_VERSION = app_settings.app_version


async def run_sse_server_async(host: str = "0.0.0.0", port: int = 8080) -> int:
    """Run MCP server with SSE transport.

    SSE transport allows:
    - Remote access over HTTP
    - Browser-based MCP clients
    - Multiple concurrent clients
    """
    from starlette.applications import Starlette
    from starlette.routing import Mount, Route
    from starlette.responses import JSONResponse
    import uvicorn

    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)
    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request):
        """Handle SSE connection from client."""
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
        return None

    async def health_check(request):
        """Health check endpoint."""
        return JSONResponse({
            "status": "ok",
            "server": SERVER_NAME,
            "version": SERVER_VERSION,
            "transport": "sse",
        })

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse_transport.handle_post_message),
            Route("/health", endpoint=health_check),
        ]
    )

    logger.info(f"Starting MCP SSE server on {host}:{port}")
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    uvicorn_server = uvicorn.Server(config)
    await uvicorn_server.serve()
    return 0


def run_sse_server(host: str = None, port: int = None) -> int:
    """Run SSE server with config from environment."""
    host = host or os.getenv("MCP_SSE_HOST", "0.0.0.0")
    port = port or int(os.getenv("MCP_SSE_PORT", "8080"))
    return asyncio.run(run_sse_server_async(host, port))


def run_server() -> int:
    """Run MCP server with transport selected by environment variable.

    MCP_TRANSPORT=stdio (default) -> local CLI mode
    MCP_TRANSPORT=sse -> remote HTTP+SSE mode
    """
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()

    if transport == "sse":
        logger.info("Starting MCP server in SSE mode")
        return run_sse_server()
    else:
        logger.info("Starting MCP server in stdio mode")
        from mcp_server.server import run_stdio_server
        return run_stdio_server()


if __name__ == "__main__":
    sys.exit(run_server())
