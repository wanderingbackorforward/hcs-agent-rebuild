"""Decision explainer — builds user-facing explanations for agent routing.

Maps internal routing decisions to human-readable text + safe metadata.
All output is filtered through mask_sensitive() before reaching the client.

Decision paths covered:
  switch         — L1 switch-word intercept
  clarify        — L2 low-confidence clarification
  lock_hit       — L3 context lock reuse
  continuation   — L3 semantic continuation
  classified     — full LLM classification result
  low_confidence — classification below threshold
"""
from typing import Any

from config.audit import mask_sensitive

# Whitelist of fields safe to expose to the client.
# Anything not in this set is stripped before emission.
_SAFE_FIELDS = {
    "intent_type", "confidence", "reason", "agent",
    "context_lock_status", "score", "stage",
}

# Human-readable explanation templates keyed by decision path.
_TEMPLATES = {
    "switch": "检测到话题切换关键词，重新分类意图",
    "clarify_multi": "输入同时涉及多个意图，需要澄清",
    "clarify_vague": "输入信息不足，需要补充细节",
    "lock_hit": "延续上轮对话，复用已有意图：{intent}",
    "continuation": "语义相似度 {score}，判定为延续对话",
    "classified": "意图分类：{intent}（置信度 {confidence:.0%}），路由到 {agent}",
    "low_confidence": "意图不明确（置信度 {confidence:.0%}），需要澄清",
}


def build_decision(
    path: str,
    intent_type: str = "",
    confidence: float = 0.0,
    agent: str = "",
    score: float = 0.0,
    context_lock_status: str = "",
    **extra: Any,
) -> dict[str, Any]:
    """Build a user-facing decision explanation dict.

    Returns ``{explanation, intent_type, confidence, reason, agent,
    context_lock_status}`` with sensitive fields stripped.
    """
    tmpl = _TEMPLATES.get(path, "")
    explanation = tmpl.format(
        intent=intent_type, confidence=confidence,
        agent=agent, score=score,
    ) if tmpl else path

    raw = {
        "explanation": explanation,
        "intent_type": intent_type,
        "confidence": round(confidence, 3),
        "reason": path,
        "agent": agent,
        "context_lock_status": context_lock_status,
        "score": round(score, 3) if score else None,
        **extra,
    }
    # Strip non-whitelisted fields + mask any sensitive values.
    safe = mask_sensitive({
        k: v for k, v in raw.items()
        if k in _SAFE_FIELDS or k == "explanation"
    })
    # Remove falsy values (empty strings, 0.0, None) to keep payload
    # compact — but always keep "explanation" and "reason".
    return {
        k: v for k, v in safe.items()
        if k in ("explanation", "reason") or v
    }


def agent_display_name(intent_type: str) -> str:
    """Map internal intent type to user-facing agent name."""
    return {
        "environment_match": "环境匹配 Agent",
        "knowledge_qa": "知识问答 Agent",
        "unrelated": "通用回复",
    }.get(intent_type, intent_type or "未知")
