"""NLI (Natural Language Inference) confidence validator.

Pluggable post-selection gate: LLM selects an agent, then NLI independently
checks whether the user query actually matches that agent's responsibility
description.  NLI does NOT participate in the selection decision — it only
provides an objective confidence score to catch LLM self-scoring errors.

Design principles:
  - Reuses SemanticChecker's embedder + INTENT_DESCRIPTIONS (zero new deps).
  - Returns None when embedder unavailable → caller falls back to v1 logic.
  - One global switch (ENABLE_NLI) to disable entirely.
"""
import logging
import math
from typing import Dict, List, Optional

from config.settings import app_settings

logger = logging.getLogger(__name__)

# Reuse the same intent descriptions as SemanticChecker for consistency.
# Extended with an "unrelated" fallback description.
AGENT_DESCRIPTIONS: Dict[str, str] = {
    "environment_match": (
        "查询匹配HCS测试环境 环境类型 组件 节点 服务状态 端口探测 资源可用性 "
        "找环境 推荐环境 筛选环境 确认状态"
    ),
    "knowledge_qa": (
        "SDK文档 用户手册 技术规范 测试规范 接口说明 安装配置 部署许可证 "
        "怎么安装 如何初始化 版本要求 部署阶段 准备阶段 验收阶段"
    ),
    "unrelated": (
        "天气 外卖 笑话 电影 与HCS测试无关的日常话题"
    ),
}

# NLI score thresholds (configurable via env).
NLI_PASS_THRESHOLD = 0.7      # >= pass → route directly
NLI_BORDERLINE_THRESHOLD = 0.6  # >= borderline but < pass → route with caution
# < borderline → fallback / clarify


class NLIValidator:
    """Pluggable NLI confidence checker.

    Uses embedding cosine similarity between user query and the selected
    agent's responsibility description as a proxy for NLI entailment score.

    When the embedder is not configured (or ENABLE_NLI is False), nli_check()
    returns None, signalling the caller to use the degradation path (LLM
    self-confidence + keyword rules).
    """

    def __init__(self, embedder=None, descriptions: Optional[Dict[str, str]] = None):
        self._embedder = embedder
        self._descriptions = descriptions or AGENT_DESCRIPTIONS
        self._desc_cache: Dict[str, List[float]] = {}

    @property
    def is_available(self) -> bool:
        """True only when NLI is enabled AND embedder is configured."""
        if not app_settings.enable_nli:
            return False
        return self._embedder is not None

    async def nli_check(self, premise: str, hypothesis_agent: str) -> Optional[float]:
        """Check entailment confidence: does `premise` (user query) match
        the responsibility of `hypothesis_agent`?

        Args:
            premise: The user's query text.
            hypothesis_agent: The agent selected by LLM (e.g. "environment_match").

        Returns:
            Float 0.0-1.0 entailment confidence, or None if NLI is unavailable
            (caller should use degradation path).
        """
        if not self.is_available:
            return None

        ref_text = self._descriptions.get(hypothesis_agent)
        if not ref_text:
            logger.warning("No description for agent '%s', NLI skip", hypothesis_agent)
            return None

        try:
            ref_emb = await self._get_desc_embedding(hypothesis_agent, ref_text)
            query_emb = await self._embedder.aembed_query(premise)
            score = _cosine_similarity(query_emb, ref_emb)
            logger.debug(
                "NLI score=%.3f agent=%s query=%s",
                score, hypothesis_agent, premise[:40],
            )
            return score
        except Exception as e:
            logger.warning("NLI check failed for agent '%s': %s", hypothesis_agent, e)
            return None

    async def _get_desc_embedding(self, agent: str, text: str) -> List[float]:
        """Cache agent description embeddings to avoid repeated API calls."""
        if agent not in self._desc_cache:
            self._desc_cache[agent] = await self._embedder.aembed_query(text)
        return self._desc_cache[agent]


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
