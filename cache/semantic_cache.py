"""Semantic cache - reuses results for similar queries using embedding similarity.

Instead of exact hash matching, uses embedding cosine similarity to find
cached queries that are semantically similar. Requires an embedder.

Interview talking point: "Semantic cache uses embedding similarity (>0.92
threshold) to reuse results for similar but not identical queries, with TTL
to avoid stale data."
"""
import time
import logging
from typing import Optional, List, Tuple
import math

from config.settings import app_settings

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = app_settings.semantic_cache_threshold
DEFAULT_TTL = app_settings.semantic_cache_ttl


class SemanticCache:
    """Semantic similarity-based cache using embeddings."""

    def __init__(self, embedder=None, ttl: int = DEFAULT_TTL,
                 threshold: float = SIMILARITY_THRESHOLD):
        self.embedder = embedder
        self.ttl = ttl
        self.threshold = threshold
        # Store: list of (embedding, query, response, timestamp)
        self._entries: List[Tuple[List[float], str, str, float]] = []

    def _embed(self, text: str) -> List[float]:
        """Embed text, adapting LangChain Embeddings and custom embedders.

        LangChain's Embeddings expose embed_query/embed_documents, while some
        custom embedders expose .embed(). This makes SemanticCache work with
        any embedder returned by create_embedding_model().
        """
        if hasattr(self.embedder, "embed_query"):
            return self.embedder.embed_query(text)
        if hasattr(self.embedder, "embed_documents"):
            return self.embedder.embed_documents([text])[0]
        return self.embedder.embed(text)

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def get(self, query: str) -> Optional[str]:
        """Find semantically similar cached response."""
        if not self.embedder or not self._entries:
            return None

        try:
            query_embedding = self._embed(query)
        except Exception as e:
            logger.warning(f"Embedding failed for semantic cache: {e}")
            return None

        now = time.time()
        best_score = 0.0
        best_response = None

        for embedding, cached_query, response, timestamp in self._entries:
            if now - timestamp >= self.ttl:
                continue
            score = self._cosine_similarity(query_embedding, embedding)
            if score > best_score:
                best_score = score
                best_response = response

        if best_score >= self.threshold:
            logger.info(f"Semantic cache hit (score={best_score:.3f}): {query[:50]}")
            return best_response
        return None

    def set(self, query: str, response: str):
        """Cache a response with its embedding."""
        if not self.embedder:
            return
        try:
            embedding = self._embed(query)
            self._entries.append((embedding, query, response, time.time()))
            # Evict expired entries.
            now = time.time()
            self._entries = [
                e for e in self._entries
                if now - e[3] < self.ttl
            ]
        except Exception as e:
            logger.warning(f"Semantic cache set failed: {e}")

    def clear(self):
        self._entries.clear()
