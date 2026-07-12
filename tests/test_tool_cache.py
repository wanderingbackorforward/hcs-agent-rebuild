"""Tests for ToolCache, the tool-cache registry, SparseRetriever (FTS5),
and HybridSearch result caching.

ToolCache and HybridSearch tests use fakes — no real ChromaDB or embedding
service needed. SparseRetriever tests use a real FTS5Store with a temp SQLite
database to verify FTS5 full-text search behavior.
"""
import os
import tempfile

import pytest
from sqlalchemy import create_engine

from cache.tool_cache import ToolCache
from rag.query_engine.hybrid_search import HybridSearch
from rag.query_engine.sparse_retriever import SparseRetriever
from rag.ingestion.storage.fts_store import FTS5Store
import cache.registry as registry


# --- reset the process-level tool cache singleton between tests ---

@pytest.fixture(autouse=True)
def _reset_tool_cache(monkeypatch):
    monkeypatch.setattr(registry, "_tool_cache", None)
    yield
    monkeypatch.setattr(registry, "_tool_cache", None)


# --- ToolCache basics ---

def test_tool_cache_get_set():
    cache = ToolCache()
    assert cache.get("q", "tool") is None
    cache.set("q", "tool", "result")
    assert cache.get("q", "tool") == "result"


def test_tool_cache_keyed_by_kwargs():
    """Same query + different top_k/filters must not collide."""
    cache = ToolCache()
    cache.set("q", "hybrid_search", "r5", top_k=5, filters=None)
    cache.set("q", "hybrid_search", "r10", top_k=10, filters=None)
    assert cache.get("q", "hybrid_search", top_k=5, filters=None) == "r5"
    assert cache.get("q", "hybrid_search", top_k=10, filters=None) == "r10"


def test_tool_cache_keyed_by_filters():
    cache = ToolCache()
    cache.set("q", "t", "scoped", filters={"category": "spec"})
    cache.set("q", "t", "unscoped", filters=None)
    assert cache.get("q", "t", filters={"category": "spec"}) == "scoped"
    assert cache.get("q", "t", filters=None) == "unscoped"


def test_tool_cache_invalidate_all():
    cache = ToolCache()
    cache.set("q", "t", "r")
    cache.invalidate_all()
    assert cache.get("q", "t") is None


def test_tool_cache_ttl_expiry(monkeypatch):
    cache = ToolCache(ttl=100)
    fake_time = [1000.0]
    monkeypatch.setattr("cache.tool_cache.time.time", lambda: fake_time[0])
    cache.set("q", "t", "r")
    assert cache.get("q", "t") == "r"
    fake_time[0] = 2000.0
    assert cache.get("q", "t") is None


# --- registry ---

def test_registry_tool_cache_singleton():
    c1 = registry.get_tool_cache()
    c2 = registry.get_tool_cache()
    assert c1 is c2


def test_registry_invalidate_tool_cache():
    cache = registry.get_tool_cache()
    cache.set("q", "t", "r")
    assert cache.get("q", "t") == "r"
    registry.invalidate_tool_cache()
    assert cache.get("q", "t") is None
    # instance still reusable
    assert registry.get_tool_cache() is cache


# --- FTS5-based SparseRetriever tests ---

@pytest.fixture
def fts_retriever():
    """Create a SparseRetriever with a real FTS5Store in a temp SQLite DB."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    engine = create_engine(f"sqlite:///{path}")
    fts_store = FTS5Store(engine=engine)

    # Insert test chunks with Chinese + English mixed content
    fts_store.insert_chunk("c1", "doc1", "HCS SDK 安装配置指南", "sdk", "SDK文档", "seed")
    fts_store.insert_chunk("c2", "doc1", "Python 版本要求 3.9 以上", "sdk", "SDK文档", "seed")
    fts_store.insert_chunk("c3", "doc2", "MySQL Redis Kafka 环境要求", "spec", "测试规范", "seed")
    fts_store.insert_chunk("c4", "doc2", "组件 available 状态检查", "spec", "测试规范", "seed")
    fts_store.insert_chunk("c5", "doc3", "部署阶段 准备 安装 验收", "manual", "部署手册", "seed")

    yield SparseRetriever(fts_store=fts_store)

    engine.dispose()
    os.unlink(path)


def test_sparse_retrieve_returns_relevant_results(fts_retriever):
    """FTS5 search returns relevant chunks for keyword queries."""
    results = fts_retriever.retrieve("SDK 安装", top_k=3)
    assert len(results) > 0
    doc_ids = {meta.get("doc_id") for _, _, _, meta in results}
    assert "doc1" in doc_ids


def test_sparse_retrieve_respects_top_k(fts_retriever):
    results = fts_retriever.retrieve("环境", top_k=2)
    assert len(results) <= 2


def test_sparse_retrieve_with_category_filter(fts_retriever):
    results = fts_retriever.retrieve("环境", top_k=5, filters={"category": "spec"})
    assert len(results) > 0
    for _, _, _, meta in results:
        assert meta.get("category") == "spec"


def test_sparse_refresh_is_noop(fts_retriever):
    """refresh() is a no-op for FTS5 — index is always up-to-date."""
    fts_retriever.refresh()  # should not raise
    results = fts_retriever.retrieve("SDK", top_k=1)
    assert len(results) > 0


def test_sparse_retrieve_empty_query_returns_empty(fts_retriever):
    results = fts_retriever.retrieve("   ", top_k=5)
    assert results == []


def test_sparse_retrieve_returns_text_and_score(fts_retriever):
    """Results must include chunk text and a numeric score."""
    results = fts_retriever.retrieve("Python", top_k=1)
    assert len(results) > 0
    chunk_id, text, score, meta = results[0]
    assert isinstance(text, str) and len(text) > 0
    assert isinstance(score, float)
    assert "doc_id" in meta


# --- fakes for HybridSearch result caching ---

class _FakeSparse:
    def __init__(self):
        self.refresh_calls = 0
        self.retrieve_calls = 0

    def refresh(self):
        self.refresh_calls += 1

    def retrieve(self, query, top_k=5, filters=None):
        self.retrieve_calls += 1
        return [("id1", "text1", 0.5, {"doc_id": "id1"})]


class _FakeDense:
    def __init__(self):
        self.retrieve_calls = 0

    def retrieve(self, query, top_k=5, filters=None):
        self.retrieve_calls += 1
        return [("id1", "text1", 0.9, {"doc_id": "id1"})]


class _FakeReranker:
    def rerank(self, query, results, top_k=5):
        return results[:top_k]


# --- HybridSearch result caching ---

def test_hybrid_search_caches_results():
    """Second search with same params must hit ToolCache, skipping refresh+retrieve."""
    sparse = _FakeSparse()
    dense = _FakeDense()
    hs = HybridSearch(
        dense=dense, sparse=sparse, reranker=_FakeReranker()
    )

    r1 = hs.search("query", top_k=5)
    r2 = hs.search("query", top_k=5)

    assert r1 == r2
    # First search ran the pipeline once; second hit cache.
    assert sparse.refresh_calls == 1, "refresh should run once, not twice"
    assert sparse.retrieve_calls == 1
    assert dense.retrieve_calls == 1


def test_hybrid_search_cache_miss_on_different_params():
    """Different top_k must miss the cache and re-run the pipeline."""
    sparse = _FakeSparse()
    dense = _FakeDense()
    hs = HybridSearch(
        dense=dense, sparse=sparse, reranker=_FakeReranker()
    )

    hs.search("query", top_k=5)
    hs.search("query", top_k=10)  # different top_k -> miss

    assert sparse.retrieve_calls == 2
    assert dense.retrieve_calls == 2
