"""MCP Tool: list_collections - lists available knowledge document collections."""
import json
import logging
from typing import Any, Dict

from mcp import types

from services.knowledge_service import KnowledgeService
from mcp_server.errors import MCPError, format_error

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
        "include_categories": {
            "type": "boolean",
            "description": "Whether to include category summaries.",
            "default": True,
        },
        "include_doc_samples": {
            "type": "boolean",
            "description": "Whether to include sample documents for each collection/category.",
            "default": False,
        },
        "sample_size": {
            "type": "integer",
            "description": "Maximum number of sample documents to include per collection.",
            "default": 3,
            "minimum": 1,
            "maximum": 20,
            "examples": [3],
        },
        "keyword": {
            "type": "string",
            "description": "Optional keyword filter for collection/category/doc title matching.",
            "examples": ["sdk", "spec"],
        },
    },
    "required": [],
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


def _doc_to_sample(doc: Any, fallback_category: str = None) -> Dict[str, Any]:
    if isinstance(doc, str):
        return {
            "doc_id": doc,
            "title": doc,
            "category": fallback_category,
        }
    return {
        "doc_id": getattr(doc, "doc_id", None) or getattr(doc, "title", None) or str(doc),
        "title": getattr(doc, "title", None) or getattr(doc, "doc_id", None) or str(doc),
        "category": getattr(doc, "category", None) or fallback_category,
    }


async def list_collections_handler(
    include_stats: bool = True,
    include_categories: bool = True,
    include_doc_samples: bool = False,
    sample_size: int = 3,
    keyword: str = "",
) -> types.CallToolResult:
    try:
        if sample_size < 1 or sample_size > 20:
            raise MCPError(
                error_type="invalid_input",
                message="sample_size must be between 1 and 20",
                details={"sample_size": sample_size},
            )

        service = KnowledgeService()
        service.initialize()
        docs = service.list_documents()
        categories = service.db.knowledge.list_categories()

        normalized_keyword = (keyword or "").strip().lower()
        category_entries = []
        matched_docs = list(docs)

        for cat in categories:
            cat_docs = service.db.knowledge.list_all(category=cat)
            cat_samples = [_doc_to_sample(doc, fallback_category=cat) for doc in cat_docs]

            if normalized_keyword:
                cat_match = normalized_keyword in cat.lower()
                sample_match = any(
                    normalized_keyword in (sample.get("doc_id") or "").lower()
                    or normalized_keyword in (sample.get("title") or "").lower()
                    for sample in cat_samples
                )
                if not cat_match and not sample_match:
                    continue

            entry: Dict[str, Any] = {"name": cat}
            if include_stats:
                entry["doc_count"] = len(cat_docs)
            if include_doc_samples:
                entry["sample_docs"] = cat_samples[:sample_size]
            category_entries.append(entry)

        if normalized_keyword:
            matched_docs = [
                doc for doc in docs
                if normalized_keyword in str(doc).lower()
            ]

        collection_entry: Dict[str, Any] = {"name": "hcs_knowledge"}
        if include_stats:
            collection_entry["doc_count"] = len(matched_docs)
        if include_categories:
            collection_entry["categories"] = category_entries
        if include_doc_samples:
            collection_entry["sample_docs"] = [
                _doc_to_sample(doc) for doc in matched_docs[:sample_size]
            ]

        payload: Dict[str, Any] = {
            "collections": [collection_entry],
            "total_collections": 1,
            "keyword_used": keyword or None,
            "include_stats": include_stats,
            "include_categories": include_categories,
            "include_doc_samples": include_doc_samples,
        }
        return _json_result(payload)
    except Exception as e:
        err = format_error(e, context="list_collections")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=err.to_text())],
            isError=True,
        )


def register_tool(protocol_handler) -> None:
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=list_collections_handler,
    )
