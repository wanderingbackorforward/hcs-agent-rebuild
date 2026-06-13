"""Reranker factory: pluggable post-retrieval reranking.

This MVP ships a NoOpReranker (returns results unchanged) so the slot is
established and tests pass. To enable a real reranker:

  1. `pip install sentence-transformers`
  2. Add CROSS_ENCODER_MODEL env var
  3. Implement CrossEncoderReranker below and register in create_reranker()
"""
import os
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional

RerankItem = Tuple[str, str, float, Dict]  # (doc_id, text, score, meta)


class Reranker(ABC):
    @abstractmethod
    def rerank(self, query: str, items: List[RerankItem], top_k: int) -> List[RerankItem]:
        """Return top_k items reordered by relevance to query."""


class NoOpReranker(Reranker):
    """Default: return first top_k items unchanged. Lets hybrid_search call
    rerank without checking for None and keeps tests green."""

    def rerank(self, query: str, items: List[RerankItem], top_k: int) -> List[RerankItem]:
        return items[:top_k]


_limiter_model = os.getenv("CROSS_ENCODER_MODEL", "").strip()


def create_reranker(model: Optional[str] = None) -> Reranker:
    """Build a Reranker.

    If CROSS_ENCODER_MODEL is set, attempt to instantiate a cross-encoder reranker;
    otherwise fall back to NoOpReranker so the system still works.
    """
    target_model = model or _limiter_model
    if not target_model:
        return NoOpReranker()
    try:
        # Lazy import so tests/dev env without sentence-transformers still work.
        from sentence_transformers import CrossEncoder
        return _CrossEncoderReranker(CrossEncoder(target_model))
    except ImportError:
        return NoOpReranker()


class _CrossEncoderReranker(Reranker):
    def __init__(self, model):
        self.model = model

    def rerank(self, query: str, items: List[RerankItem], top_k: int) -> List[RerankItem]:
        if not items:
            return []
        pairs = [(query, text) for _, text, _, _ in items]
        scores = self.model.predict(pairs).tolist()
        scored = list(zip(items, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scored[:top_k]]
