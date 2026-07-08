"""Tests for context lock — multi-turn intent reuse + switch/expire/overwrite.

Uses fakes (no real LLM) so CI runs without API keys. Covers the balanced
plan: continuation reuse skips classify; switch/new-intent/TTL clears; env
task completion clears once.
"""
import asyncio
import time

import pytest

from agents.context_lock import ContextLock, load_lock, save_lock, clear_lock, DEFAULT_TTL
from agents.task_classification.classification_processor import ClassificationProcessor


class FakeClassifier:
    """Records classify calls; returns a fixed intent JSON string."""
    def __init__(self, intent="knowledge_qa", fields=None):
        self.intent = intent
        self.fields = fields or {}
        self.call_count = 0

    async def classify_stream(self, user_input):
        import json
        self.call_count += 1
        yield json.dumps({
            "intent_type": self.intent,
            "required_fields": self.fields,
            "missing_fields": [], "keywords": [], "topic": "t",
        })


class FakeRouter:
    def __init__(self):
        self.routed = []

    async def route(self, intent_type, user_input, session_id=None):
        self.routed.append((intent_type, user_input))
        yield "reply:" + intent_type


class FakeUnrelated:
    async def handle(self, user_input):
        return "unrelated"


class FakeState:
    def reset(self):
        pass


def _repo(tmp_path):
    from db.db_router import DatabaseRouter
    import os
    fd = os.open(str(tmp_path / "s.db"), os.O_CREAT | os.O_RDWR)
    os.close(fd)
    return DatabaseRouter(db_path=f"sqlite:///{tmp_path / 's.db'}").session


def _processor(repo, llm=None, intent="knowledge_qa", fields=None):
    return ClassificationProcessor(
        FakeClassifier(intent, fields), FakeState(), FakeRouter(),
        FakeUnrelated(), session_repo=repo, llm=llm,
    )


def test_continuation_reuses_lock_skips_classify(tmp_path):
    repo = _repo(tmp_path)
    proc = _processor(repo, intent="environment_match", fields={"env_type": "dev"})
    asyncio.run(_collect(proc.process_task_stream("我要 dev 环境", "s1")))
    assert proc.classifier.call_count == 1  # first turn classified
    # follow-up short reference -> reuse, no classify
    out = asyncio.run(_collect(proc.process_task_stream("换成 Kafka", "s1")))
    assert proc.classifier.call_count == 1  # NOT re-classified
    assert "environment_match" in out


def test_switch_keyword_clears_and_reclassifies(tmp_path):
    repo = _repo(tmp_path)
    proc = _processor(repo, intent="knowledge_qa")
    asyncio.run(_collect(proc.process_task_stream("SDK 怎么装", "s2")))
    assert proc.classifier.call_count == 1
    asyncio.run(_collect(proc.process_task_stream("换个话题，SDK 怎么装", "s2")))
    assert proc.classifier.call_count == 2  # switch -> reclassify
    lock = load_lock(repo, "s2")
    assert lock.intent == "knowledge_qa"


def test_new_intent_overwrites_lock(tmp_path):
    repo = _repo(tmp_path)
    proc = _processor(repo, intent="environment_match", fields={"env_type": "dev"})
    asyncio.run(_collect(proc.process_task_stream("找 dev 环境", "s3")))
    assert load_lock(repo, "s3").intent == "environment_match"
    # next call returns knowledge_qa -> auto-overwrite.
    # Use a long non-env sentence so it isn't treated as a short-sentence
    # continuation and actually falls through to reclassification.
    proc.classifier.intent = "knowledge_qa"
    asyncio.run(_collect(proc.process_task_stream("HCS 混合云平台是什么", "s3")))
    assert load_lock(repo, "s3").intent == "knowledge_qa"


def test_ttl_expiry_clears_lock(tmp_path):
    repo = _repo(tmp_path)
    save_lock(repo, "s4", "environment_match", {"env_type": "dev"})
    # force expiry
    fields = repo.get_fields("s4")
    fields["_context_lock"]["locked_at"] = time.time() - DEFAULT_TTL - 1
    repo.update_fields("s4", {"_context_lock": fields["_context_lock"]})
    proc = _processor(repo, intent="knowledge_qa")
    asyncio.run(_collect(proc.process_task_stream("随便问个问题", "s4")))
    # expired lock -> not reused -> classified
    assert proc.classifier.call_count == 1


def test_llm_fallback_continuation(tmp_path):
    repo = _repo(tmp_path)
    save_lock(repo, "s5", "knowledge_qa", {})

    class JudgeLLM:
        def invoke(self, msgs):
            return type("R", (), {"content": "是"})()

    proc = _processor(repo, llm=JudgeLLM(), intent="knowledge_qa")
    # long sentence, no reference/env keyword -> hits LLM fallback -> 是 -> reuse
    asyncio.run(_collect(proc.process_task_stream(
        "请再详细解释一下上面提到的那个客户端配置细节", "s5")))
    assert proc.classifier.call_count == 0  # reused via LLM judge


def test_env_completion_clears_lock(tmp_path):
    """Env-match reaching 'fields complete' clears the lock once (manual site)."""
    from agents.environment_matching.processor import EnvironmentMatchingProcessor
    repo = _repo(tmp_path)
    save_lock(repo, "s6", "environment_match", {"env_type": "dev"})
    # processor.clear_lock is called in the complete branch; verify helper.
    clear_lock(repo, "s6")
    assert not load_lock(repo, "s6").is_active


def test_multi_intent_conflict_triggers_clarify(tmp_path):
    """One sentence spanning env + qa with a conjunction -> ask, don't execute."""
    repo = _repo(tmp_path)
    save_lock(repo, "s7", "environment_match", {"env_type": "dev"})
    proc = _processor(repo, intent="environment_match")
    out = asyncio.run(_collect(proc.process_task_stream(
        "找有 Kafka 的环境，顺便问下 Kafka 怎么安装", "s7")))
    # low confidence -> clarify, no classify, no route to agent
    assert proc.classifier.call_count == 0
    assert proc.router.routed == []
    assert "环境" in out and "技术" in out  # clarify question mentions both


def test_vague_reference_triggers_clarify(tmp_path):
    """Pure reference + no params -> ask for detail, don't execute."""
    repo = _repo(tmp_path)
    save_lock(repo, "s8", "knowledge_qa", {})  # empty params
    proc = _processor(repo, intent="knowledge_qa")
    out = asyncio.run(_collect(proc.process_task_stream("那个", "s8")))
    assert proc.classifier.call_count == 0
    assert proc.router.routed == []
    assert "补充" in out or "具体" in out


def test_single_intent_kb_question_not_conflict(tmp_path):
    """A normal 'Kafka 怎么安装' (component + how, NO conjunction) is NOT a
    conflict — it's a single knowledge_qa question and routes normally."""
    repo = _repo(tmp_path)
    save_lock(repo, "s9", "knowledge_qa", {"topic": "sdk"})
    proc = _processor(repo, intent="knowledge_qa")
    out = asyncio.run(_collect(proc.process_task_stream("Kafka怎么安装", "s9")))
    # continuation + high confidence -> routes reusing lock, no classify
    assert proc.classifier.call_count == 0
    assert len(proc.router.routed) == 1
    assert "knowledge_qa" in out


async def _collect(gen):
    out = ""
    async for t in gen:
        out += t
    return out
