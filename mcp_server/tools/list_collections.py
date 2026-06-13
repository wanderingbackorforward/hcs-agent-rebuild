"""MCP Tool: list_collections - lists available knowledge document collections."""
import logging
from typing import Any, Dict

from mcp import types

from services.knowledge_service import KnowledgeService

logger = logging.getLogger(__name__)

TOOL_NAME = "list_collections"
TOOL_DESCRIPTION = "List all available document collections in the HCS knowledge base."
TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "include_stats": {
            "type": "boolean",
            "description": "Whether to include document counts.",
            "default": True,
        },
    },
    "required": [],
}


async def list_collections_handler(include_stats: bool = True) -> types.CallToolResult:
    try:
        service = KnowledgeService()
        service.initialize()
        docs = service.list_documents()
        categories = service.db.knowledge.list_categories()
        lines = ["## Available Collections", ""]
        lines.append(f"- **hcs_knowledge** - {len(docs)} documents")
        lines.append("")
        lines.append("## SQLite Categories")
        for cat in categories:
            count = len(service.db.knowledge.list_all(category=cat))
            lines.append(f"- {cat}: {count} documents")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="\n".join(lines))],
            isError=False,
        )
    except Exception as e:
        logger.exception("list_collections failed")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Error: {e}")],
            isError=True,
        )


def register_tool(protocol_handler) -> None:
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=list_collections_handler,
    )
