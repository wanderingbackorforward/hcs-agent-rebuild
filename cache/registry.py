"""Process-level cache registry — singletons shared across sessions.

Caches are created lazily on first use so import-time failures in the
embedding client don't break application startup. All KnowledgeQAAgent
instances in the process share one SemanticCache, maximizing the hit rate.

When REDIS_URL is set, caches are backed by Redis (shared across processes
and restarts). Without Redis, they fall back to process-local in-memory
storage. Either way, the registry API is identical.

Interview talking point: "The semantic cache is a process-level singleton
behind a registry — every session's agent hits the same cache, so a
question answered once serves all future askers. The embedder is created
lazily and degrades to no-op if the embedding service is unavailable."
"""
import logging

from cache.semantic_cache import SemanticCache, create_semantic_cache
from cache.tool_cache import ToolCache, create_tool_cache
from config.settings import app_settings

logger = logging.getLogger(__name__)

_semantic_cache: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    """Return the process-wide semantic cache, creating it lazily.

    The embedder is created on first call via create_embedding_model(). If
    that fails (no key / unsupported model), the cache degrades to a no-op
    (embedder=None) so the agent path still works, just uncached.

    The backend (Redis vs in-memory) is chosen by create_semantic_cache()
    based on whether REDIS_URL is configured.
    """
    global _semantic_cache
    if _semantic_cache is None:
        try:
            from config.model_provider import create_embedding_model
            embedder = create_embedding_model()
        except Exception as e:
            logger.warning("Semantic cache init failed, degrading to no-op: %s", e)
            embedder = None
        _semantic_cache = create_semantic_cache(
            embedder=embedder,
            ttl=app_settings.semantic_cache_ttl,
            threshold=app_settings.semantic_cache_threshold,
            max_entries=app_settings.semantic_cache_max_entries,
        )
        if embedder is not None:
            logger.info(
                "Semantic cache initialized (threshold=%.2f, ttl=%ds)",
                _semantic_cache.threshold, _semantic_cache.ttl,
            )
    return _semantic_cache


def invalidate_semantic_cache() -> None:
    """Clear the semantic cache — call after knowledge base updates.

    Called by KnowledgeService.ingest_text so newly ingested documents
    produce fresh answers instead of returning stale cached ones.

    Works with both Redis and in-memory backends (both implement clear()).
    """
    global _semantic_cache
    if _semantic_cache is not None:
        _semantic_cache.clear()
        logger.info("Semantic cache cleared (knowledge base update)")


_tool_cache: ToolCache | None = None


def get_tool_cache() -> ToolCache:
    """Return the process-wide tool cache, creating it lazily.

    Caches retrieval results (HybridSearch output) by query hash so repeated
    queries skip the dense+BM25+RRF+rerank pipeline. Invalidated on document
    updates via invalidate_tool_cache().

    The backend (Redis vs in-memory) is chosen by create_tool_cache()
    based on whether REDIS_URL is configured.
    """
    global _tool_cache
    if _tool_cache is None:
        _tool_cache = create_tool_cache()
    return _tool_cache


def invalidate_tool_cache() -> None:
    """Clear the tool cache — call after knowledge base updates.

    Works with both Redis and in-memory backends (both implement
    invalidate_all()).
    """
    global _tool_cache
    if _tool_cache is not None:
        _tool_cache.invalidate_all()
        logger.info("Tool cache cleared (knowledge base update)")
