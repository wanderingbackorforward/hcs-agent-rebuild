"""MCP Resource: knowledge_catalog — exposes the knowledge base document index.

This is a *data-exposure* resource: the LLM can read it to discover what
documents and collections exist before deciding which Tool to call.
It complements the list_collections Tool (which is an *action*) by
providing a lightweight, always-available catalog the LLM can browse
without triggering a full tool invocation.
"""
import logging

from services.knowledge_service import KnowledgeService
from config.audit import sanitize_text

logger = logging.getLogger(__name__)

RESOURCE_URI = "hcs://knowledge/catalog"
RESOURCE_NAME = "knowledge-catalog"
RESOURCE_DESCRIPTION = (
    "Index of all documents and collections in the HCS knowledge base. "
    "Read this to discover available knowledge before querying."
)
RESOURCE_MIME_TYPE = "text/plain"


def knowledge_catalog_handler() -> str:
    """Return a human-readable catalog of knowledge documents."""
    try:
        service = KnowledgeService()
        service.initialize()
        docs = service.list_documents()
        categories = service.db.knowledge.list_categories()

        lines = ["# HCS Knowledge Catalog", ""]
        lines.append(f"**Total documents:** {len(docs)}")
        lines.append(f"**Categories:** {', '.join(categories) if categories else 'none'}")
        lines.append("")
        lines.append("## Documents")
        for doc_id in docs:
            summary = service.get_document_summary(doc_id)
            preview = (summary or "")[:80].replace("\n", " ")
            lines.append(f"- **{doc_id}**: {preview}...")
        return "\n".join(lines)
    except Exception as e:
        safe_msg = sanitize_text(str(e))
        logger.warning(f"knowledge_catalog resource failed: {safe_msg}")
        return f"# HCS Knowledge Catalog\n\n(Unable to load catalog: {safe_msg})"


def register_resource(protocol_handler) -> None:
    protocol_handler.register_resource(
        uri=RESOURCE_URI,
        name=RESOURCE_NAME,
        description=RESOURCE_DESCRIPTION,
        handler=knowledge_catalog_handler,
        mime_type=RESOURCE_MIME_TYPE,
    )
