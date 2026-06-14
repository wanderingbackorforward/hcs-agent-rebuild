"""Reranker factory tests. Cover NoOp default + factory wiring + a fake
CrossEncoder to validate the integration without requiring the
sentence-transformers download."""
import os
from typing import List, Tuple, Dict

import pytest

from config import reranker_factory as rf
from config.reranker_factory import (
    RerankItem,
    NoOpReranker,
    create_reranker,
    _CrossEncoderReranker,
)


def test_default_is_noop_when_env_unset(monkeypatch):
    monkeypatch.delenv("CROSS_ENCODER_MODEL", raising=False)
    r = create_reranker()
    assert isinstance(r, NoOpReranker)


def test_falls_back_to_noop_when_lib_missing(monkeypatch):
    """When CROSS_ENCODER_MODEL is set but sentence-transformers is not
    installed, the factory should log a warning and return NoOpReranker
    rather than crash."""
    monkeypatch.setenv("CROSS_ENCODER_MODEL", "BAAI/bge-reranker-base")

    # Force the ImportError branch by monkeypatching the import.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("simulated missing dep")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    r = create_reranker()
    assert isinstance(r, NoOpReranker)


class _FakeCrossEncoder:
    """Deterministic stand-in: scores each pair = length of text. Tests
    can then assert that longer-text items are reordered above shorter ones.

    Returns a numpy array (mimicking real sentence-transformers API)."""

    def predict(self, pairs: List[Tuple[str, str]]):
        try:
            import numpy as np
            return np.array([float(len(text)) for _, text in pairs], dtype=float)
        except ImportError:
            # Fallback: return a list-like that supports .tolist()
            class _ListWithToList(list):
                def tolist(self):
                    return list(self)
            return _ListWithToList([float(len(text)) for _, text in pairs])


def test_cross_encoder_rerank_reorders_by_score():
    r = _CrossEncoderReranker(_FakeCrossEncoder())
    items: List[RerankItem] = [
        ("a", "hi", 0.1, {}),
        ("b", "a much longer piece of text content here", 0.5, {}),
        ("c", "medium length text", 0.3, {}),
    ]
    out = r.rerank(query="q", items=items, top_k=2)
    assert len(out) == 2
    # FakeCrossEncoder scores by text length, so order should be b, c
    assert out[0][0] == "b"
    assert out[1][0] == "c"


def test_cross_encoder_rerank_empty():
    r = _CrossEncoderReranker(_FakeCrossEncoder())
    assert r.rerank("q", [], top_k=5) == []


def test_noop_rerank_truncates_to_topk():
    r = NoOpReranker()
    items: List[RerankItem] = [
        (f"d{i}", f"text {i}", float(i), {"i": i}) for i in range(10)
    ]
    out = r.rerank("q", items, top_k=3)
    assert len(out) == 3
    assert [d[0] for d in out] == ["d0", "d1", "d2"]
