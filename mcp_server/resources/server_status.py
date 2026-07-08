"""MCP Resource: server_status — exposes MCP server runtime metadata.

Resources are *passive*: the LLM discovers them via resources/list and
reads them via resources/read. Unlike Tools, the LLM never decides to
"call" a resource — it simply reads the data when it needs context.

This resource exposes server name, version, and counts of registered
tools / resources / prompts, giving the LLM a snapshot of what the
server can do without listing every schema.
"""
import json
from typing import Any, Dict


RESOURCE_URI = "hcs://server/status"
RESOURCE_NAME = "server-status"
RESOURCE_DESCRIPTION = (
    "Runtime status of the HCS MCP server: name, version, and counts "
    "of registered tools, resources, and prompts."
)
RESOURCE_MIME_TYPE = "application/json"


def server_status_handler(protocol_handler) -> str:
    """Return a JSON snapshot of the server's registered primitives."""

    def _inner() -> str:
        data: Dict[str, Any] = {
            "server_name": protocol_handler.server_name,
            "server_version": protocol_handler.server_version,
            "tools": list(protocol_handler.tools.keys()),
            "tool_count": len(protocol_handler.tools),
            "resources": list(protocol_handler.resources.keys()),
            "resource_count": len(protocol_handler.resources),
            "prompts": list(protocol_handler.prompts.keys()),
            "prompt_count": len(protocol_handler.prompts),
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    return _inner


def register_resource(protocol_handler) -> None:
    """Register this resource with the protocol handler.

    The handler closure captures a reference to protocol_handler so it
    can report live counts at read time (not registration time).
    """
    protocol_handler.register_resource(
        uri=RESOURCE_URI,
        name=RESOURCE_NAME,
        description=RESOURCE_DESCRIPTION,
        handler=server_status_handler(protocol_handler),
        mime_type=RESOURCE_MIME_TYPE,
    )
