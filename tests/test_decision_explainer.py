"""Tests for decision explainer module."""
import pytest

from config.decision_explainer import build_decision, agent_display_name


class TestAgentDisplayName:
    def test_known_intents(self):
        assert agent_display_name("environment_match") == "环境匹配 Agent"
        assert agent_display_name("knowledge_qa") == "知识问答 Agent"
        assert agent_display_name("unrelated") == "通用回复"

    def test_unknown_intent(self):
        assert agent_display_name("custom") == "custom"

    def test_empty_intent(self):
        assert agent_display_name("") == "未知"


class TestBuildDecision:
    def test_switch_path(self):
        d = build_decision("switch")
        assert d["explanation"] == "检测到话题切换关键词，重新分类意图"
        assert d["reason"] == "switch"
        assert "intent_type" not in d  # None stripped

    def test_classified_path(self):
        d = build_decision(
            "classified", intent_type="knowledge_qa",
            confidence=0.92, agent="知识问答 Agent",
        )
        assert "知识问答 Agent" in d["explanation"]
        assert d["intent_type"] == "knowledge_qa"
        assert d["confidence"] == 0.92
        assert d["agent"] == "知识问答 Agent"
        assert "92%" in d["explanation"]

    def test_low_confidence_path(self):
        d = build_decision(
            "low_confidence", intent_type="environment_match",
            confidence=0.35,
        )
        assert "35%" in d["explanation"]
        assert d["confidence"] == 0.35

    def test_lock_hit_path(self):
        d = build_decision(
            "lock_hit", intent_type="knowledge_qa",
            context_lock_status="active",
        )
        assert "延续上轮" in d["explanation"]
        assert d["context_lock_status"] == "active"

    def test_clarify_multi_path(self):
        d = build_decision("clarify_multi", score=0.2)
        assert "多个意图" in d["explanation"]
        assert d["score"] == 0.2

    def test_clarify_vague_path(self):
        d = build_decision("clarify_vague", score=0.3)
        assert "不足" in d["explanation"]

    def test_none_values_stripped(self):
        d = build_decision("classified", intent_type="knowledge_qa")
        # confidence, agent, score, context_lock_status should be absent
        assert "confidence" not in d
        assert "agent" not in d
        assert "score" not in d

    def test_extra_fields_not_in_safe_list_are_dropped(self):
        """Only whitelisted fields + explanation are exposed."""
        d = build_decision(
            "classified", intent_type="knowledge_qa",
            confidence=0.9, agent="KB Agent",
            internal_prompt="secret system prompt",  # should be dropped
            api_key="sk-12345",  # should be dropped
        )
        assert "internal_prompt" not in d
        assert "api_key" not in d
        assert d["intent_type"] == "knowledge_qa"

    def test_confidence_rounded(self):
        d = build_decision("classified", confidence=0.95678, agent="X")
        assert d["confidence"] == 0.957

    def test_continuation_path(self):
        d = build_decision("continuation", score=0.88)
        assert "0.88" in d["explanation"]
        assert d["score"] == 0.88
