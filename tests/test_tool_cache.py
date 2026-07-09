"""Tests for ToolCache, the tool-cache registry, SparseRetriever refresh-skip,
and HybridSearch result caching.

All tests use fakes — no real ChromaDB or embedding service needed.
"""
import pytest

from cache.tool_cache import ToolCache
from rag.query_engine.hybrid_search import HybridSearch
from rag.query_engine.sparse_retriever import SparseRetriever
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


# --- fakes for SparseRetriever / HybridSearch ---

class _FakeCollection:
    """Mimics chromadb Collection.get/count without a real DB."""

    def __init__(self, docs, metas, ids):
        self._docs = list(docs)
        self._metas = list(metas)
        self._ids = list(ids)
        self.get_calls = 0

    def get(self, include=None):
        self.get_calls += 1
        return {
            "documents": list(self._docs),
            "metadatas": list(self._metas),
            "ids": list(self._ids),
        }

    def count(self):
        return len(self._docs)


class _FakeStore:
    def __init__(self, collection):
        self.collection = collection


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


# --- SparseRetriever refresh-skip ---

def test_sparse_refresh_skips_when_count_unchanged():
    col = _FakeCollection(["doc one two"], [{"doc_id": "d1"}], ["id1"])
    sparse = SparseRetriever(store=_FakeStore(col))
    assert col.get_calls == 1  # initial _build_index

    sparse.refresh()  # count unchanged -> must skip rebuild
    assert col.get_calls == 1, "refresh must not rebuild when count is unchanged"


def test_sparse_refresh_rebuilds_when_count_changes():
    col = _FakeCollection(["doc one"], [{"doc_id": "d1"}], ["id1"])
    sparse = SparseRetriever(store=_FakeStore(col))
    assert col.get_calls == 1

    col._docs.append("new doc")  # simulate ingest -> count changes
    sparse.refresh()
    assert col.get_calls == 2, "refresh must rebuild when count changed"


def test_sparse_retrieve_uses_cached_ids():
    """retrieve() must NOT hit collection.get — it should use self._ids."""
    col = _FakeCollection(["hello world"], [{"doc_id": "d1"}], ["chunk_abc"])
    sparse = SparseRetriever(store=_FakeStore(col))
    get_calls_after_build = col.get_calls

    results = sparse.retrieve("hello", top_k=1)
    assert col.get_calls == get_calls_after_build, "retrieve must not call collection.get"
    assert results[0][0] == "chunk_abc"  # cached id used, not a fallback


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
