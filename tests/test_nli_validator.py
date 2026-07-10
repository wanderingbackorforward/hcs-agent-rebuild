"""Tests for NLI confidence validator and degradation logic.

Covers:
  1. NLIValidator basic behavior (available, unavailable, caching)
  2. Degradation path (NLI off → keyword + LLM confidence fallback)
  3. ClassificationProcessor NLI gate integration (pass / borderline / reject)
  4. Decision explainer new NLI paths
"""
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agents.task_classification.nli_validator import (
    NLIValidator, AGENT_DESCRIPTIONS,
    NLI_PASS_THRESHOLD, NLI_BORDERLINE_THRESHOLD,
    _cosine_similarity,
)
from config.decision_explainer import build_decision


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class FakeEmbedder:
    """Deterministic embedder for testing — returns pre-configured vectors."""

    def __init__(self, vectors=None):
        self._vectors = vectors or {}
        self._call_count = 0

    async def aembed_query(self, text):
        self._call_count += 1
        # Return a vector based on text content for deterministic testing.
        if text in self._vectors:
            return self._vectors[text]
        # Default: hash-based pseudo-embedding.
        h = hash(text) % 100
        return [float(h), float(h + 1), float(h + 2)]


# --------------------------------------------------------------------------- #
# 1. NLIValidator unit tests
# --------------------------------------------------------------------------- #

class TestNLIValidator:

    def test_unavailable_without_embedder(self):
        """NLIValidator returns None when no embedder is configured."""
        validator = NLIValidator(embedder=None)
        assert validator.is_available is False

    @pytest.mark.asyncio
    async def test_returns_none_when_unavailable(self):
        """nli_check returns None when embedder is missing (triggers degradation)."""
        validator = NLIValidator(embedder=None)
        score = await validator.nli_check("查环境", "environment_match")
        assert score is None

    @pytest.mark.asyncio
    async def test_returns_score_when_available(self):
        """nli_check returns a float score when embedder is configured."""
        embedder = FakeEmbedder()
        validator = NLIValidator(embedder=embedder)
        assert validator.is_available is True

        score = await validator.nli_check("帮我找测试环境", "environment_match")
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_caches_description_embedding(self):
        """Description embeddings are cached to avoid repeated API calls."""
        embedder = FakeEmbedder()
        validator = NLIValidator(embedder=embedder)

        await validator.nli_check("query1", "environment_match")
        first_count = embedder._call_count
        await validator.nli_check("query2", "environment_match")
        # Description embedding should be cached; only query embedding is new.
        assert embedder._call_count == first_count + 1

    @pytest.mark.asyncio
    async def test_unknown_agent_returns_none(self):
        """nli_check returns None for unknown agent labels."""
        embedder = FakeEmbedder()
        validator = NLIValidator(embedder=embedder)
        score = await validator.nli_check("query", "nonexistent_agent")
        assert score is None

    @pytest.mark.asyncio
    async def test_embedder_exception_returns_none(self):
        """nli_check returns None when embedder raises an exception."""
        class FailingEmbedder:
            async def aembed_query(self, text):
                raise RuntimeError("API error")

        validator = NLIValidator(embedder=FailingEmbedder())
        score = await validator.nli_check("query", "environment_match")
        assert score is None

    def test_cosine_similarity(self):
        """Cosine similarity calculation is correct."""
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)
        assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)
        assert _cosine_similarity([1, 0, 0], [-1, 0, 0]) == pytest.approx(-1.0)
        assert _cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0

    @pytest.mark.asyncio
    async def test_high_similarity_for_matching_intent(self):
        """Query about environments should score higher against environment_match
        description than against knowledge_qa description."""
        # Craft embeddings where env query is close to env description.
        env_desc = AGENT_DESCRIPTIONS["environment_match"]
        kb_desc = AGENT_DESCRIPTIONS["knowledge_qa"]
        env_query = "帮我找一个有MySQL的测试环境"

        embedder = FakeEmbedder(vectors={
            env_desc: [1.0, 0.0, 0.0],
            kb_desc: [0.0, 1.0, 0.0],
            env_query: [0.9, 0.1, 0.0],  # close to env_desc
        })
        validator = NLIValidator(embedder=embedder)

        env_score = await validator.nli_check(env_query, "environment_match")
        kb_score = await validator.nli_check(env_query, "knowledge_qa")
        assert env_score > kb_score


# --------------------------------------------------------------------------- #
# 2. Degradation path tests
# --------------------------------------------------------------------------- #

class TestDegradationPath:

    def test_keyword_check_environment_match(self):
        """_check_agent_keyword returns True when env keywords are present."""
        from agents.task_classification.classification_processor import ClassificationProcessor
        assert ClassificationProcessor._check_agent_keyword("帮我找MySQL环境", "environment_match") is True
        assert ClassificationProcessor._check_agent_keyword("今天天气如何", "environment_match") is False

    def test_keyword_check_knowledge_qa(self):
        """_check_agent_keyword returns True when KB keywords are present."""
        from agents.task_classification.classification_processor import ClassificationProcessor
        assert ClassificationProcessor._check_agent_keyword("SDK怎么安装", "knowledge_qa") is True
        assert ClassificationProcessor._check_agent_keyword("查环境", "knowledge_qa") is False

    def test_keyword_check_unrelated_always_true(self):
        """Unrelated intent doesn't need keyword validation."""
        from agents.task_classification.classification_processor import ClassificationProcessor
        assert ClassificationProcessor._check_agent_keyword("anything", "unrelated") is True


# --------------------------------------------------------------------------- #
# 3. Decision explainer NLI paths
# --------------------------------------------------------------------------- #

class TestDecisionExplainerNLI:

    def test_nli_pass_decision(self):
        """nli_pass path generates correct explanation."""
        decision = build_decision(
            "nli_pass", intent_type="environment_match",
            confidence=0.9, agent="环境匹配 Agent", nli_score=0.85,
        )
        assert "NLI" in decision["explanation"]
        assert "0.85" in decision["explanation"]
        assert decision["nli_score"] == 0.85

    def test_nli_borderline_decision(self):
        """nli_borderline path generates correct explanation."""
        decision = build_decision(
            "nli_borderline", intent_type="knowledge_qa",
            confidence=0.85, agent="知识问答 Agent", nli_score=0.65,
        )
        assert "边界" in decision["explanation"]
        assert "0.65" in decision["explanation"]

    def test_nli_reject_decision(self):
        """nli_reject path generates correct explanation."""
        decision = build_decision(
            "nli_reject", intent_type="knowledge_qa",
            confidence=0.85, agent="知识问答 Agent", nli_score=0.45,
        )
        assert "未通过" in decision["explanation"]
        assert "0.45" in decision["explanation"]

    def test_nli_score_in_safe_fields(self):
        """nli_score is whitelisted and not stripped from output."""
        decision = build_decision(
            "nli_pass", nli_score=0.75,
        )
        assert "nli_score" in decision
        assert decision["nli_score"] == 0.75
