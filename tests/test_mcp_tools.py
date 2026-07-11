"""MCP tool handler unit tests.

Use direct in-process invocation (no MCP stdio transport) to keep tests fast.
These tests are true unit tests: they patch the tool modules with fake
KnowledgeService / fake LLM so they do not depend on SQLite writes, vector
seeding, or external model keys.
"""
import json
import pytest

class _Chunk:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    async def astream(self, _messages):
        yield _Chunk("这是 ")
        yield _Chunk("生成答案")


class FakeDoc:
    def __init__(
        self,
        content: str,
        doc_id: str = None,
        title: str = None,
        category: str = None,
        source: str = None,
        metadata_json: dict = None,
    ):
        self.content = content
        self.doc_id = doc_id
        self.title = title
        self.category = category
        self.source = source
        self.metadata_json = metadata_json or {}


class FakeKnowledgeRepo:
    def __init__(self, categories=None, docs_by_category=None, docs_by_id=None):
        self._categories = categories or []
        self._docs_by_category = docs_by_category or {}
        self._docs_by_id = docs_by_id or {}

    def list_categories(self):
        return list(self._categories)

    def list_all(self, category=None):
        return list(self._docs_by_category.get(category, []))

    def get_by_doc_id(self, doc_id):
        raw = self._docs_by_id.get(doc_id)
        if raw is None:
            return None
        if isinstance(raw, FakeDoc):
            return raw
        if isinstance(raw, dict):
            return FakeDoc(
                content=raw.get("content", ""),
                doc_id=doc_id,
                title=raw.get("title"),
                category=raw.get("category"),
                source=raw.get("source"),
                metadata_json=raw.get("metadata_json"),
            )
        return FakeDoc(content=raw, doc_id=doc_id, title=doc_id)


class FakeDB:
    def __init__(self, knowledge_repo):
        self.knowledge = knowledge_repo


class FakeKnowledgeService:
    def __init__(self, *, results=None, docs=None, summary_map=None, categories=None, docs_by_category=None, docs_by_id=None):
        self._results = results or []
        self._docs = docs or []
        self._summary_map = summary_map or {}
        self.db = FakeDB(
            FakeKnowledgeRepo(
                categories=categories,
                docs_by_category=docs_by_category,
                docs_by_id=docs_by_id,
            )
        )

    def initialize(self):
        return None

    def search(self, _query, top_k=None, filters=None):
        results = self._results
        if filters and "doc_id" in filters:
            results = [
                item for item in results
                if item[3].get("doc_id") == filters["doc_id"]
            ]
        if filters and "category" in filters:
            results = [
                item for item in results
                if item[3].get("category") == filters["category"]
            ]
        return results[:top_k] if top_k else results

    def list_documents(self):
        return list(self._docs)

    def get_document_summary(self, doc_id):
        return self._summary_map.get(doc_id)


@pytest.fixture(autouse=True)
def clear_tool_cache():
    from cache.registry import invalidate_tool_cache
    invalidate_tool_cache()
    yield
    invalidate_tool_cache()


def test_query_knowledge_hub_happy_path(monkeypatch):
    from mcp_server.tools import query_knowledge_hub
    import asyncio
    fake_service = FakeKnowledgeService(
        results=[
            (
                "chunk-1",
                "HCS SDK 安装命令是 pip install hcs-sdk。",
                0.1234,
                {
                    "doc_id": "hcs-sdk-quickstart",
                    "title": "HCS SDK 快速入门",
                    "category": "sdk",
                    "source": "seed",
                    "chunk_index": 0,
                },
            )
        ]
    )
    monkeypatch.setattr(query_knowledge_hub, "KnowledgeService", lambda: fake_service)
    monkeypatch.setattr(query_knowledge_hub, "create_chat_model", lambda temperature=0: FakeLLM())
    res = asyncio.run(
        query_knowledge_hub.query_knowledge_hub_handler(
            query="HCS SDK 怎么安装", top_k=3, return_mode="both", category="sdk",
        )
    )
    assert res.isError is False
    payload = json.loads(res.content[0].text)
    assert payload["query"] == "HCS SDK 怎么安装"
    assert payload["top_k_used"] == 3
    assert payload["return_mode"] == "both"
    assert payload["result_count"] >= 1
    assert isinstance(payload["retrieved_chunks"], list)
    assert payload["answer_generated_by"] == "llm"
    assert payload["filters_used"]["category"] == "sdk"
    assert payload["answer"] == "这是 生成答案"
    assert payload["retrieved_chunks"][0]["doc_id"] == "hcs-sdk-quickstart"
    assert payload["retrieved_chunks"][0]["metadata"]["chunk_index"] == 0


def test_query_knowledge_hub_no_results(monkeypatch):
    """Unknown term still returns a non-error result, not an exception."""
    from mcp_server.tools import query_knowledge_hub
    import asyncio
    monkeypatch.setattr(
        query_knowledge_hub,
        "KnowledgeService",
        lambda: FakeKnowledgeService(results=[]),
    )
    res = asyncio.run(
        query_knowledge_hub.query_knowledge_hub_handler(
            query="xyz完全不相关的话题xyz",
        )
    )
    # Should be a clean not-found or empty-result, not a raw exception.
    assert res.isError is False
    payload = json.loads(res.content[0].text)
    assert payload["result_count"] == 0
    assert payload["message"] == "未找到相关资料。"


def test_query_knowledge_hub_invalid_return_mode():
    from mcp_server.tools import query_knowledge_hub
    import asyncio
    res = asyncio.run(
        query_knowledge_hub.query_knowledge_hub_handler(
            query="HCS 是什么",
            return_mode="invalid-mode",
        )
    )
    assert res.isError is True
    text = res.content[0].text
    assert "error_type=invalid_input" in text


def test_get_document_summary_not_found(monkeypatch):
    from mcp_server.tools import get_document_summary
    import asyncio
    fake_service = FakeKnowledgeService(summary_map={}, docs_by_id={})
    monkeypatch.setattr(get_document_summary, "KnowledgeService", lambda: fake_service)
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


def test_get_document_summary_happy_path(monkeypatch):
    from mcp_server.tools import get_document_summary
    import asyncio
    fake_service = FakeKnowledgeService(
        summary_map={"hcs-sdk-quickstart": "HCS SDK 是华为混合云平台的开发工具包。"},
        docs_by_id={
            "hcs-sdk-quickstart": {
                "content": "HCS SDK 是华为混合云平台的开发工具包。",
                "title": "HCS SDK 快速入门",
                "category": "sdk",
                "source": "seed",
                "metadata_json": {"chunk_count": 4, "owner": "docs-team"},
            }
        },
    )
    monkeypatch.setattr(get_document_summary, "KnowledgeService", lambda: fake_service)
    res = asyncio.run(
        get_document_summary.get_document_summary_handler(
            doc_id="hcs-sdk-quickstart",
            max_chars=200,
            include_metadata=True,
            include_source=True,
            include_chunk_stats=True,
        )
    )
    assert res.isError is False
    payload = json.loads(res.content[0].text)
    assert payload["doc_id"] == "hcs-sdk-quickstart"
    assert payload["title"] == "HCS SDK 快速入门"
    assert payload["category"] == "sdk"
    assert payload["source"] == "seed"
    assert payload["chunk_count"] == 4
    assert payload["metadata"]["owner"] == "docs-team"
    assert "HCS SDK" in payload["summary"]


def test_get_document_summary_invalid_max_chars():
    from mcp_server.tools import get_document_summary
    import asyncio
    res = asyncio.run(
        get_document_summary.get_document_summary_handler(
            doc_id="hcs-sdk-quickstart",
            max_chars=50,
        )
    )
    assert res.isError is True
    assert "error_type=invalid_input" in res.content[0].text


def test_list_collections_returns_counts(monkeypatch):
    from mcp_server.tools import list_collections
    import asyncio
    fake_service = FakeKnowledgeService(
        docs=["hcs-sdk-quickstart", "hcs-test-spec-env", "hcs-manual-deploy"],
        categories=["sdk", "spec", "manual"],
        docs_by_category={
            "sdk": ["hcs-sdk-quickstart"],
            "spec": ["hcs-test-spec-env"],
            "manual": ["hcs-manual-deploy"],
        },
    )
    monkeypatch.setattr(list_collections, "KnowledgeService", lambda: fake_service)
    res = asyncio.run(
        list_collections.list_collections_handler(
            include_stats=True,
            include_categories=True,
            include_doc_samples=True,
            sample_size=2,
        )
    )
    assert res.isError is False
    payload = json.loads(res.content[0].text)
    assert payload["total_collections"] == 1
    assert payload["include_doc_samples"] is True
    assert payload["collections"][0]["name"] == "hcs_knowledge"
    assert payload["collections"][0]["doc_count"] == 3
    assert len(payload["collections"][0]["categories"]) == 3
    assert payload["collections"][0]["categories"][0]["doc_count"] == 1
    assert len(payload["collections"][0]["sample_docs"]) == 2


def test_list_collections_keyword_filter(monkeypatch):
    from mcp_server.tools import list_collections
    import asyncio
    fake_service = FakeKnowledgeService(
        docs=["hcs-sdk-quickstart", "hcs-test-spec-env", "hcs-manual-deploy"],
        categories=["sdk", "spec", "manual"],
        docs_by_category={
            "sdk": ["hcs-sdk-quickstart"],
            "spec": ["hcs-test-spec-env"],
            "manual": ["hcs-manual-deploy"],
        },
    )
    monkeypatch.setattr(list_collections, "KnowledgeService", lambda: fake_service)
    res = asyncio.run(
        list_collections.list_collections_handler(
            include_stats=True,
            include_categories=True,
            include_doc_samples=True,
            keyword="sdk",
        )
    )
    assert res.isError is False
    payload = json.loads(res.content[0].text)
    assert payload["keyword_used"] == "sdk"
    assert payload["collections"][0]["doc_count"] == 1
    assert len(payload["collections"][0]["categories"]) == 1
    assert payload["collections"][0]["categories"][0]["name"] == "sdk"


def test_list_collections_invalid_sample_size():
    from mcp_server.tools import list_collections
    import asyncio
    res = asyncio.run(
        list_collections.list_collections_handler(sample_size=0)
    )
    assert res.isError is True
    assert "error_type=invalid_input" in res.content[0].text
