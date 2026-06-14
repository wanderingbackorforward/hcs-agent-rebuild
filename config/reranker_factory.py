"""Reranker factory: pluggable post-retrieval reranking.

Selection:
  - NoOpReranker (default): pass-through, no extra dep
  - CrossEncoderReranker: when CROSS_ENCODER_MODEL env var is set AND
    sentence-transformers is installed

Recommended models (multilingual, Chinese-friendly):
  - cross-encoder/mmarco-mMiniLMv2-L12-H384-v1  (~100MB, fast, good baseline)
  - BAAI/bge-reranker-v2-m3                    (~600MB, best quality for zh+en)
  - BAAI/bge-reranker-base                     (~300MB, good speed/quality)

Usage:
  # No rerank (default; tests, dev):
  unset CROSS_ENCODER_MODEL

  # Real rerank (production):
  pip install sentence-transformers
  export CROSS_ENCODER_MODEL=BAAI/bge-reranker-v2-m3
  # first call downloads the model (~600MB) into HF cache
"""
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Dict

logger = logging.getLogger(__name__)

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


_env_model = os.getenv("CROSS_ENCODER_MODEL", "").strip()


def create_reranker(model: Optional[str] = None) -> Reranker:
    """Build a Reranker.

    If CROSS_ENCODER_MODEL is set, attempt to instantiate a cross-encoder
    reranker; otherwise fall back to NoOpReranker so the system still works.
    """
    target_model = model or _env_model
    if not target_model:
        return NoOpReranker()
    try:
        # Lazy import so tests/dev env without sentence-transformers still work.
        from sentence_transformers import CrossEncoder
        return _CrossEncoderReranker(CrossEncoder(target_model))
    except ImportError:
        logger.warning(
            "CROSS_ENCODER_MODEL=%r set but sentence-transformers not installed; "
            "falling back to NoOpReranker. Install with: pip install sentence-transformers",
            target_model,
        )
        return NoOpReranker()


class _CrossEncoderReranker(Reranker):
    """Real cross-encoder reranker. Slower than NoOp (~50ms per pair) but
    substantially improves precision on borderline results."""

    def __init__(self, model: Any):
        self.model = model

    def rerank(self, query: str, items: List[RerankItem], top_k: int) -> List[RerankItem]:
        if not items:
            return []
        pairs = [(query, text) for _, text, _, _ in items]
        scores = self.model.predict(pairs).tolist()
        scored = list(zip(items, scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [it for it, _ in scored[:top_k]]


# Public re-export so tests can import via either name.
CrossEncoderReranker = _CrossEncoderReranker
