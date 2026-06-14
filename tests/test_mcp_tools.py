"""MCP tool handler unit tests.

Use direct in-process invocation (no MCP stdio transport) to keep tests fast.
Each handler is exercised for: success path, input validation, and the
format_error envelope.

These tests need an embedding-capable environment to seed the knowledge
base with real vectors. When no key is set (CI's default state), they
skip — the hybrid_search test (test_hybrid_search.py) is the dedicated
embedding-API integration test.
"""
import os
import tempfile
import pytest

from db.db_router import DatabaseRouter
from services.knowledge_service import KnowledgeService


def _has_embedding_key() -> bool:
    return bool(
        os.getenv("LLM_API_KEY")
        or os.getenv("EMBEDDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )


# Module-level skipif: tests need real embeddings to seed the KB.
requires_embedding = pytest.mark.skipif(
    not _has_embedding_key(),
    reason="no LLM/Embedding API key set; KB seeding requires it",
)


@pytest.fixture
def knowledge_service(monkeypatch):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db_router = DatabaseRouter(db_path=f"sqlite:///{path}")
    svc = KnowledgeService(db_router)
    svc.initialize()
    yield svc
    db_router.close()
    os.unlink(path)


@requires_embedding
def test_query_knowledge_hub_happy_path(knowledge_service):
    from mcp_server.tools import query_knowledge_hub
    import asyncio
    res = asyncio.run(
        query_knowledge_hub.query_knowledge_hub_handler(
            query="HCS SDK 怎么安装", top_k=3,
        )
    )
    assert res.isError is False
    text = res.content[0].text
    assert "References" in text or "AI Answer" in text


@requires_embedding
def test_query_knowledge_hub_no_results(knowledge_service):
    """Unknown term still returns a non-error result, not an exception."""
    from mcp_server.tools import query_knowledge_hub
    import asyncio
    res = asyncio.run(
        query_knowledge_hub.query_knowledge_hub_handler(
            query="xyz完全不相关的话题xyz",
        )
    )
    # Should be a clean not-found or empty-result, not a raw exception.
    assert res.isError is False


@requires_embedding
def test_get_document_summary_not_found(knowledge_service):
    from mcp_server.tools import get_document_summary
    import asyncio
    res = asyncio.run(
        get_document_summary.get_document_summary_handler(doc_id="does-not-exist")
    )
    assert res.isError is True
    text = res.content[0].text
    # Must use structured error envelope, never raw exception.
    assert "error_type=not_found" in text
    assert "trace_id=" in text
    # And NO raw exception leakage patterns.
    assert "Traceback" not in text
    assert "sqlite" not in text.lower()  # no internal info


@requires_embedding
def test_get_document_summary_happy_path(knowledge_service):
    from mcp_server.tools import get_document_summary
    import asyncio
    res = asyncio.run(
        get_document_summary.get_document_summary_handler(doc_id="hcs-sdk-quickstart")
    )
    assert res.isError is False
    text = res.content[0].text
    assert "HCS SDK" in text


@requires_embedding
def test_list_collections_returns_counts(knowledge_service):
    from mcp_server.tools import list_collections
    import asyncio
    res = asyncio.run(list_collections.list_collections_handler(include_stats=True))
    assert res.isError is False
    text = res.content[0].text
    assert "Available Collections" in text
    # 3 seed docs
    assert "3 documents" in text
