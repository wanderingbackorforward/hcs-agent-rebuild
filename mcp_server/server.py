"""MCP Server entry point using official MCP SDK (stdio transport)."""
import asyncio
import sys

from config.settings import app_settings
from mcp_server.protocol_handler import create_mcp_server

SERVER_NAME = app_settings.mcp_server_name
SERVER_VERSION = app_settings.app_version


def _redirect_all_loggers_to_stderr():
    import logging as _logging
    root = _logging.getLogger()
    stderr_handler = _logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(
        _logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    for handler in root.handlers[:]:
        if isinstance(handler, _logging.StreamHandler) and not isinstance(handler, _logging.FileHandler):
            root.removeHandler(handler)
    root.addHandler(stderr_handler)


async def run_stdio_server_async() -> int:
    import mcp.server.stdio
    _redirect_all_loggers_to_stderr()
    server = create_mcp_server(SERVER_NAME, SERVER_VERSION)
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )
    return 0


def run_stdio_server() -> int:
    return asyncio.run(run_stdio_server_async())


def main() -> int:
    """Run MCP server with transport selected by environment variable.

    MCP_TRANSPORT=stdio (default) -> local CLI mode
    MCP_TRANSPORT=sse -> remote HTTP+SSE mode
    """
    import os
    transport = os.getenv("MCP_TRANSPORT", "stdio").strip().lower()
    if transport == "sse":
        from mcp_server.sse_server import run_sse_server
        return run_sse_server()
    return run_stdio_server()


if __name__ == "__main__":
    sys.exit(main())
