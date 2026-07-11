"""MCP Tool: get_document_summary - retrieves summary for a specific document."""
import json
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
        "max_chars": {
            "type": "integer",
            "description": "Maximum summary length to return.",
            "default": 500,
            "minimum": 100,
            "maximum": 5000,
        },
        "include_metadata": {
            "type": "boolean",
            "description": "Whether to include document metadata.",
            "default": True,
        },
        "include_source": {
            "type": "boolean",
            "description": "Whether to include document source field.",
            "default": True,
        },
        "include_chunk_stats": {
            "type": "boolean",
            "description": "Whether to include content length and chunk statistics.",
            "default": True,
        },
    },
    "required": ["doc_id"],
}


def _json_result(payload: Dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=json.dumps(payload, ensure_ascii=False, indent=2),
            )
        ],
        isError=False,
    )


async def get_document_summary_handler(
    doc_id: str,
    collection: Optional[str] = None,
    max_chars: int = 500,
    include_metadata: bool = True,
    include_source: bool = True,
    include_chunk_stats: bool = True,
) -> types.CallToolResult:
    try:
        if max_chars < 100 or max_chars > 5000:
            raise MCPError(
                error_type="invalid_input",
                message="max_chars must be between 100 and 5000",
                details={"max_chars": max_chars},
            )
        service = KnowledgeService()
        service.initialize()
        doc = service.db.knowledge.get_by_doc_id(doc_id)
        content = service.get_document_summary(doc_id)
        if not content:
            content = doc.content if doc else None
        if not content:
            raise MCPError(
                error_type="not_found",
                message=f"Document '{doc_id}' not found.",
                details={"doc_id": doc_id, "collection": collection},
            )
        summary = content[:max_chars] + "..." if len(content) > max_chars else content
        metadata_json = getattr(doc, "metadata_json", None) or {}

        payload: Dict[str, Any] = {
            "doc_id": doc_id,
            "title": getattr(doc, "title", None) or doc_id,
            "collection": collection or "hcs_knowledge",
            "category": getattr(doc, "category", None),
            "summary": summary,
        }
        if include_source:
            payload["source"] = getattr(doc, "source", None)
        if include_chunk_stats:
            payload["content_length"] = len(content)
            payload["chunk_count"] = metadata_json.get("chunk_count")
        if include_metadata:
            payload["metadata"] = metadata_json
        return _json_result(payload)
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
