"""MCP Tool: get_document_summary - retrieves summary for a specific document."""
import logging
from typing import Any, Dict, Optional

from mcp import types

from services.knowledge_service import KnowledgeService
from mcp_server.errors import MCPError, format_error

logger = logging.getLogger(__name__)

TOOL_NAME = "get_document_summary"
TOOL_DESCRIPTION = "Get summary and metadata for a specific HCS knowledge document."
TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_id": {
            "type": "string",
            "description": "The document ID to retrieve summary for.",
        },
        "collection": {
            "type": "string",
            "description": "Collection name. Optional.",
        },
    },
    "required": ["doc_id"],
}


async def get_document_summary_handler(
    doc_id: str,
    collection: Optional[str] = None,
) -> types.CallToolResult:
    try:
        service = KnowledgeService()
        service.initialize()
        content = service.get_document_summary(doc_id)
        if not content:
            doc = service.db.knowledge.get_by_doc_id(doc_id)
            content = doc.content if doc else None
        if not content:
            raise MCPError(
                error_type="not_found",
                message=f"Document '{doc_id}' not found.",
                details={"doc_id": doc_id, "collection": collection},
            )
        summary = content[:500] + "..." if len(content) > 500 else content
        lines = [
            f"## Document: {doc_id}",
            "",
            f"**Collection:** {collection or 'hcs_knowledge'}",
            f"**Chunks/Length:** {len(content)} chars",
            "",
            "### Summary",
            summary,
        ]
        return types.CallToolResult(
            content=[types.TextContent(type="text", text="\n".join(lines))],
            isError=False,
        )
    except Exception as e:
        err = format_error(e, context="get_document_summary")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=err.to_text())],
            isError=True,
        )


def register_tool(protocol_handler) -> None:
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=get_document_summary_handler,
    )
