"""Semantic continuation checker using embedding cosine similarity.

Replaces keyword-based continuation detection with embedding similarity.
Uses the project's configured embedding model (OpenAI-compatible or
MiniMax). Falls back gracefully (returns None) when no embedder is
configured, letting the caller fall back to keyword rules or LLM judge.
"""
import logging
import math
from typing import Optional

from agents.context_lock import ContextLock

logger = logging.getLogger(__name__)

# Domain descriptions for each intent — embedded once and cached.
# These act as "prototype" texts: if the user's input is semantically
# close to the domain description, it's likely a continuation.
INTENT_DESCRIPTIONS = {
    "environment_match": "查询匹配HCS测试环境 环境类型 组件 节点 服务状态 端口探测 资源可用性",
    "knowledge_qa": "SDK文档 用户手册 技术规范 测试规范 接口说明 安装配置 部署许可证",
}

DEFAULT_SIMILARITY_THRESHOLD = 0.65


class SemanticChecker:
    """Check continuation via embedding cosine similarity.

    The embedder is the same model used for RAG / ChromaDB, so no extra
    dependency is needed.  Intent-description embeddings are cached on
    first use to avoid repeated API calls.
    """

    def __init__(self, embedder=None, threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self._embedder = embedder
        self._threshold = threshold
        self._desc_cache: dict[str, list[float]] = {}

    @property
    def is_available(self) -> bool:
        return self._embedder is not None

    async def check_continuation(self, text: str, lock: ContextLock) -> Optional[bool]:
        """Return True if text is semantically close to the locked intent.

        Returns None if the checker is unavailable (caller should fall
        back to keyword rules or LLM judge).
        """
        if not self._embedder or not lock.intent:
            return None

        ref_text = INTENT_DESCRIPTIONS.get(lock.intent)
        if not ref_text:
            return None

        try:
            ref_emb = await self._get_desc_embedding(lock.intent, ref_text)
            user_emb = await self._embedder.aembed_query(text)
            sim = _cosine_similarity(user_emb, ref_emb)
            logger.debug(
                "semantic sim=%.3f intent=%s text=%s", sim, lock.intent, text[:30]
            )
            return sim >= self._threshold
        except Exception as e:
            logger.warning("semantic continuation check failed: %s", e)
            return None

    async def _get_desc_embedding(self, intent: str, text: str) -> list[float]:
        if intent not in self._desc_cache:
            self._desc_cache[intent] = await self._embedder.aembed_query(text)
        return self._desc_cache[intent]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
