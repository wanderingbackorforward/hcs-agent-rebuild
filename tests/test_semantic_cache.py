"""Tests for SemanticCache and the cache registry.

Covers:
- _embed() adapter: works with LangChain Embeddings (embed_query /
  embed_documents) and custom embedders (.embed()).
- semantic hit (similar query reuses answer) and miss (different query).
- TTL expiry, clear, and embedder=None no-op degradation.
- registry singleton + invalidation wiring.
"""
from pathlib import Path

import pytest

from cache.semantic_cache import SemanticCache
import cache.registry as registry


# --- Fake embedders mimicking the three interface shapes ---

class _LangChainEmbedder:
    """Mimics LangChain Embeddings: has embed_query + embed_documents."""

    def __init__(self, vectors):
        self._v = vectors
        self.query_calls = 0
        self.doc_calls = 0

    def embed_query(self, text):
        self.query_calls += 1
        return list(self._v.get(text, [0.0, 0.0]))

    def embed_documents(self, texts):
        self.doc_calls += 1
        return [list(self._v.get(t, [0.0, 0.0])) for t in texts]


class _DocumentsOnlyEmbedder:
    """Only has embed_documents (no embed_query)."""

    def __init__(self, vectors):
        self._v = vectors

    def embed_documents(self, texts):
        return [list(self._v.get(t, [0.0, 0.0])) for t in texts]


class _CustomEmbedder:
    """Only has .embed() (no LangChain interface)."""

    def __init__(self, vectors):
        self._v = vectors

    def embed(self, text):
        return list(self._v.get(text, [0.0, 0.0]))


# --- _embed adapter ---

def test_embed_uses_langchain_embed_query():
    emb = _LangChainEmbedder({"x": [1.0, 0.0]})
    cache = SemanticCache(embedder=emb)
    assert cache._embed("x") == [1.0, 0.0]
    assert emb.query_calls == 1
    assert emb.doc_calls == 0  # must prefer embed_query


def test_embed_falls_back_to_embed_documents():
    emb = _DocumentsOnlyEmbedder({"x": [0.0, 1.0]})
    cache = SemanticCache(embedder=emb)
    assert cache._embed("x") == [0.0, 1.0]


def test_embed_falls_back_to_custom_embed():
    emb = _CustomEmbedder({"x": [1.0, 1.0]})
    cache = SemanticCache(embedder=emb)
    assert cache._embed("x") == [1.0, 1.0]


# --- hit / miss ---

def test_semantic_hit_same_vector_reuses_answer():
    # Two queries mapped to the same vector -> cosine similarity 1.0.
    vectors = {
        "mysql安装方法": [1.0, 0.0],
        "安装mysql的步骤": [1.0, 0.0],
    }
    cache = SemanticCache(embedder=_LangChainEmbedder(vectors), threshold=0.9)
    cache.set("mysql安装方法", "用 pip 安装")
    assert cache.get("安装mysql的步骤") == "用 pip 安装"


def test_semantic_miss_different_vector():
    vectors = {
        "mysql安装方法": [1.0, 0.0],
        "今天天气如何": [0.0, 1.0],
    }
    cache = SemanticCache(embedder=_LangChainEmbedder(vectors), threshold=0.92)
    cache.set("mysql安装方法", "用 pip 安装")
    # Orthogonal vectors -> similarity 0 -> miss.
    assert cache.get("今天天气如何") is None


def test_semantic_miss_empty_cache():
    cache = SemanticCache(embedder=_LangChainEmbedder({}))
    assert cache.get("anything") is None


def test_semantic_hit_identical_query():
    vectors = {"q": [1.0, 0.0]}
    cache = SemanticCache(embedder=_LangChainEmbedder(vectors))
    cache.set("q", "answer")
    assert cache.get("q") == "answer"


# --- no-embedder degradation ---

def test_no_embedder_is_noop():
    cache = SemanticCache(embedder=None)
    cache.set("q", "answer")  # must not raise
    assert cache.get("q") is None  # nothing cached without embeddings


# --- TTL expiry ---

def test_ttl_expiry_returns_none(monkeypatch):
    vectors = {"q": [1.0, 0.0]}
    cache = SemanticCache(embedder=_LangChainEmbedder(vectors), ttl=100)

    fake_time = [1000.0]
    monkeypatch.setattr("cache.semantic_cache.time.time", lambda: fake_time[0])

    cache.set("q", "answer")  # timestamp = 1000
    assert cache.get("q") == "answer"  # within TTL

    fake_time[0] = 2000.0  # advance past TTL
    assert cache.get("q") is None


# --- clear ---

def test_clear_empties_cache():
    vectors = {"q": [1.0, 0.0]}
    cache = SemanticCache(embedder=_LangChainEmbedder(vectors))
    cache.set("q", "answer")
    assert cache.get("q") == "answer"

    cache.clear()
    assert cache.get("q") is None
    assert cache._entries == []


# --- registry singleton + invalidation ---

@pytest.fixture
def _reset_registry(monkeypatch):
    """Reset the process-level singleton so tests are isolated."""
    monkeypatch.setattr(registry, "_semantic_cache", None)
    yield
    monkeypatch.setattr(registry, "_semantic_cache", None)


def test_registry_returns_singleton(_reset_registry, monkeypatch):
    fake = _LangChainEmbedder({"x": [1.0, 0.0]})
    monkeypatch.setattr(
        "config.model_provider.create_embedding_model", lambda: fake
    )
    c1 = registry.get_semantic_cache()
    c2 = registry.get_semantic_cache()
    assert c1 is c2, "registry must return the same instance"
    assert c1.embedder is fake


def test_registry_invalidate_clears_entries(_reset_registry, monkeypatch):
    fake = _LangChainEmbedder({"q": [1.0, 0.0]})
    monkeypatch.setattr(
        "config.model_provider.create_embedding_model", lambda: fake
    )
    cache = registry.get_semantic_cache()
    cache.set("q", "answer")
    assert len(cache._entries) == 1

    registry.invalidate_semantic_cache()
    assert cache._entries == []
    # Instance is still reusable (singleton not reset, just cleared).
    assert registry.get_semantic_cache() is cache


def test_registry_degrades_to_noop_on_failure(_reset_registry, monkeypatch):
    def _boom():
        raise RuntimeError("embedding service down")
    monkeypatch.setattr(
        "config.model_provider.create_embedding_model", _boom
    )
    cache = registry.get_semantic_cache()
    assert cache.embedder is None  # degraded
    assert cache.get("q") is None  # no-op, not a crash
