"""Process-level cache registry — singletons shared across sessions.

Caches are created lazily on first use so import-time failures in the
embedding client don't break application startup. All KnowledgeQAAgent
instances in the process share one SemanticCache, maximizing the hit rate.

Interview talking point: "The semantic cache is a process-level singleton
behind a registry — every session's agent hits the same cache, so a
question answered once serves all future askers. The embedder is created
lazily and degrades to no-op if the embedding service is unavailable."
"""
import logging

from cache.semantic_cache import SemanticCache
from cache.tool_cache import ToolCache

logger = logging.getLogger(__name__)

_semantic_cache: SemanticCache | None = None


def get_semantic_cache() -> SemanticCache:
    """Return the process-wide SemanticCache, creating it lazily.

    The embedder is created on first call via create_embedding_model(). If
    that fails (no key / unsupported model), the cache degrades to a no-op
    (embedder=None) so the agent path still works, just uncached.
    """
    global _semantic_cache
    if _semantic_cache is None:
        try:
            from config.model_provider import create_embedding_model
            _semantic_cache = SemanticCache(embedder=create_embedding_model())
            logger.info(
                "Semantic cache initialized (threshold=%.2f, ttl=%ds)",
                _semantic_cache.threshold, _semantic_cache.ttl,
            )
        except Exception as e:
            logger.warning("Semantic cache init failed, degrading to no-op: %s", e)
            _semantic_cache = SemanticCache(embedder=None)
    return _semantic_cache


def invalidate_semantic_cache() -> None:
    """Clear the semantic cache — call after knowledge base updates.

    Called by KnowledgeService.ingest_text so newly ingested documents
    produce fresh answers instead of returning stale cached ones.
    """
    global _semantic_cache
    if _semantic_cache is not None:
        _semantic_cache.clear()
        logger.info("Semantic cache cleared (knowledge base update)")


_tool_cache: ToolCache | None = None


def get_tool_cache() -> ToolCache:
    """Return the process-wide ToolCache, creating it lazily.

    Caches retrieval results (HybridSearch output) by query hash so repeated
    queries skip the dense+BM25+RRF+rerank pipeline. Invalidated on document
    updates via invalidate_tool_cache().
    """
    global _tool_cache
    if _tool_cache is None:
        _tool_cache = ToolCache()
    return _tool_cache


def invalidate_tool_cache() -> None:
    """Clear the tool cache — call after knowledge base updates."""
    global _tool_cache
    if _tool_cache is not None:
        _tool_cache.invalidate_all()
        logger.info("Tool cache cleared (knowledge base update)")
