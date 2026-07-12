"""Semantic cache - reuses results for similar queries using embedding similarity.

Instead of exact hash matching, uses embedding cosine similarity to find
cached queries that are semantically similar. Requires an embedder.

Two backends:
- InMemorySemanticCache: process-local list store with pure-Python cosine
- RedisSemanticCache: Redis HASH store with numpy vectorized cosine

The create_semantic_cache() factory picks the right backend automatically.

Interview talking point: "Semantic cache uses embedding similarity (>0.92
threshold) to reuse results for similar but not identical queries, with TTL
to avoid stale data. When Redis is available, similarity is computed in a
single vectorized numpy operation across all entries."
"""
import time
import json
import uuid
import logging
import math
from typing import Optional, List, Tuple

from config.settings import app_settings
from config.database import get_redis_client

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = app_settings.semantic_cache_threshold
DEFAULT_TTL = app_settings.semantic_cache_ttl
DEFAULT_MAX_ENTRIES = app_settings.semantic_cache_max_entries

# Import RedisError defensively so the module loads even without the redis
# package installed. When redis is missing, get_redis_client() returns None
# and RedisSemanticCache is never instantiated.
try:
    from redis.exceptions import RedisError
except ImportError:  # pragma: no cover - redis is in requirements.txt
    RedisError = Exception


class InMemorySemanticCache:
    """Semantic similarity-based cache using embeddings (in-memory).

    Stores entries as a list of (embedding, query, response, timestamp)
    tuples and computes cosine similarity with pure Python.
    """

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


class RedisSemanticCache:
    """Redis-backed semantic cache using numpy vectorized cosine similarity.

    Stores entries in a Redis HASH (key=sem_cache): field=uuid,
    value=JSON({embedding, query, response, timestamp}).
    On get(), fetches all entries, filters by TTL, and computes vectorized
    cosine similarity via numpy in a single matrix operation.

    All Redis errors degrade to None/no-op. numpy is imported lazily inside
    methods so the module loads even if numpy is somehow not installed.
    """

    HASH_KEY = "sem_cache"

    def __init__(self, redis_client, embedder=None, ttl: int = DEFAULT_TTL,
                 threshold: float = SIMILARITY_THRESHOLD,
                 max_entries: int = DEFAULT_MAX_ENTRIES):
        self.redis = redis_client
        self.embedder = embedder
        self.ttl = ttl
        self.threshold = threshold
        self.max_entries = max_entries

    def _embed(self, text: str) -> List[float]:
        """Embed text, adapting LangChain Embeddings and custom embedders."""
        if hasattr(self.embedder, "embed_query"):
            return self.embedder.embed_query(text)
        if hasattr(self.embedder, "embed_documents"):
            return self.embedder.embed_documents([text])[0]
        return self.embedder.embed(text)

    def get(self, query: str) -> Optional[str]:
        """Find semantically similar cached response in Redis."""
        if not self.embedder:
            return None

        try:
            query_embedding = self._embed(query)
        except Exception as e:
            logger.warning(f"Embedding failed for semantic cache: {e}")
            return None

        try:
            all_entries = self.redis.hgetall(self.HASH_KEY)
        except RedisError as e:
            logger.warning(f"Redis semantic cache get failed: {e}")
            return None

        if not all_entries:
            return None

        # Parse and filter by TTL.
        now = time.time()
        valid_entries = []
        for _field, value in all_entries.items():
            try:
                entry = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                continue
            timestamp = entry.get("timestamp", 0)
            if now - timestamp >= self.ttl:
                continue
            valid_entries.append(entry)

        if not valid_entries:
            return None

        # Lazy import numpy — avoids import errors if numpy is not installed.
        try:
            import numpy as np
        except ImportError:
            logger.warning(
                "numpy not installed; Redis semantic cache cannot compute similarity"
            )
            return None

        try:
            emb_matrix = np.array(
                [e["embedding"] for e in valid_entries], dtype=float
            )
            query_emb = np.array(query_embedding, dtype=float)
            # Vectorized cosine similarity.
            norms = np.linalg.norm(emb_matrix, axis=1)
            query_norm = np.linalg.norm(query_emb)
            similarities = (emb_matrix @ query_emb) / (norms * query_norm + 1e-8)
            best_idx = int(np.argmax(similarities))
            best_score = float(similarities[best_idx])
            if best_score >= self.threshold:
                logger.info(
                    f"Semantic cache hit (Redis, score={best_score:.3f}): {query[:50]}"
                )
                return valid_entries[best_idx]["response"]
        except Exception as e:
            logger.warning(
                f"Redis semantic cache similarity computation failed: {e}"
            )
            return None

        return None

    def set(self, query: str, response: str):
        """Cache a response with its embedding in Redis."""
        if not self.embedder:
            return

        try:
            embedding = self._embed(query)
        except Exception as e:
            logger.warning(f"Semantic cache set embedding failed: {e}")
            return

        entry = {
            "embedding": embedding,
            "query": query,
            "response": response,
            "timestamp": time.time(),
        }
        field = str(uuid.uuid4())
        try:
            self.redis.hset(self.HASH_KEY, field, json.dumps(entry))
            self._evict_if_needed()
        except RedisError as e:
            logger.warning(f"Redis semantic cache set failed: {e}")
        except (TypeError, ValueError) as e:
            logger.warning(f"Redis semantic cache serialization failed: {e}")

    def _evict_if_needed(self):
        """Remove oldest entries if the hash exceeds max_entries."""
        try:
            count = self.redis.hlen(self.HASH_KEY)
        except RedisError as e:
            logger.warning(f"Redis semantic cache hlen failed: {e}")
            return
        if count <= self.max_entries:
            return

        try:
            all_entries = self.redis.hgetall(self.HASH_KEY)
        except RedisError as e:
            logger.warning(f"Redis semantic cache evict hgetall failed: {e}")
            return

        # Sort by timestamp ascending, remove the oldest.
        parsed = []
        for f, v in all_entries.items():
            try:
                e = json.loads(v)
                parsed.append((f, e.get("timestamp", 0)))
            except (json.JSONDecodeError, TypeError):
                parsed.append((f, 0))
        parsed.sort(key=lambda x: x[1])
        to_remove = len(parsed) - self.max_entries
        if to_remove > 0:
            fields_to_delete = [f for f, _ in parsed[:to_remove]]
            try:
                self.redis.hdel(self.HASH_KEY, *fields_to_delete)
            except RedisError as e:
                logger.warning(f"Redis semantic cache evict hdel failed: {e}")

    def clear(self):
        """Delete the entire semantic cache hash."""
        try:
            self.redis.delete(self.HASH_KEY)
            logger.info("Semantic cache cleared (Redis)")
        except RedisError as e:
            logger.warning(f"Redis semantic cache clear failed: {e}")


def create_semantic_cache(embedder=None, ttl: Optional[int] = None,
                          threshold: Optional[float] = None,
                          max_entries: Optional[int] = None):
    """Factory: return a Redis-backed cache if Redis is available, else in-memory.

    Args:
        embedder: Embedding model instance (LangChain Embeddings or custom).
            None disables caching (no-op).
        ttl: Cache TTL in seconds. Defaults to app_settings.semantic_cache_ttl.
        threshold: Cosine similarity threshold for a cache hit.
            Defaults to app_settings.semantic_cache_threshold.
        max_entries: Max entries before oldest are evicted (Redis only).
            Defaults to app_settings.semantic_cache_max_entries.

    Returns:
        RedisSemanticCache if a Redis client is available, otherwise
        InMemorySemanticCache.
    """
    if ttl is None:
        ttl = DEFAULT_TTL
    if threshold is None:
        threshold = SIMILARITY_THRESHOLD
    if max_entries is None:
        max_entries = DEFAULT_MAX_ENTRIES
    redis_client = get_redis_client()
    if redis_client is not None:
        return RedisSemanticCache(
            redis_client, embedder, ttl, threshold, max_entries
        )
    return InMemorySemanticCache(embedder, ttl, threshold)


# Backward-compat alias: existing code/tests do ``from cache.semantic_cache import SemanticCache``.
SemanticCache = InMemorySemanticCache
