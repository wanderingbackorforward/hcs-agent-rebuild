"""Agent caching module - three-layer cache for LLM results, tool results, and semantic cache.

The semantic cache is wired into KnowledgeQAAgent via cache.registry: the
process-wide singleton is created lazily on first use and shared across all
sessions. It is invalidated automatically when new documents are ingested.
"""
from cache.llm_cache import LLMCache
from cache.tool_cache import ToolCache
from cache.semantic_cache import SemanticCache
from cache.registry import (
    get_semantic_cache,
    get_tool_cache,
    invalidate_semantic_cache,
    invalidate_tool_cache,
)

__all__ = [
    "LLMCache",
    "ToolCache",
    "SemanticCache",
    "get_semantic_cache",
    "get_tool_cache",
    "invalidate_semantic_cache",
    "invalidate_tool_cache",
]
