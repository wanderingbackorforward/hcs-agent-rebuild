"""Classification processor - orchestrates classify -> route -> respond pipeline.

Context lock (multi-turn): a follow-up turn that looks like a continuation
(short sentence / reference / same-domain keyword) reuses the locked intent
and SKIPS LLM re-classification — saving tokens and preventing intent drift.
A switch keyword, a new intent, or TTL expiry clears the lock. Judgment lives
here (entry point, one place); persistence is delegated to agents.context_lock.
"""
import json
import logging
import re
from pathlib import Path
from typing import AsyncGenerator, Optional

from config.audit import audit_event
from agents.context_lock import ContextLock, load_lock, save_lock, clear_lock

logger = logging.getLogger(__name__)

_PROMPTS = Path(__file__).parent.parent.parent / "prompts"
SWITCH_WORDS = ("换个话题", "换话题", "退出", "不查了", "问别的", "换个问题",
                "算了", "新建任务", "不查环境")
REFERENCE_WORDS = ("那个", "这个", "换成", "改成", "再加", "再来一个",
                   "上面那个", "刚才", "它的", "另一个")
ENV_WORDS = ("环境", "组件", "节点", "hbase", "kafka", "mysql", "redis",
             "区域", "region", "测试环境", "可用", "状态", "staging", "dev")


class ClassificationProcessor:
    def __init__(self, classifier, state_manager, router, unrelated_handler,
                 session_repo=None, llm=None):
        self.classifier = classifier
        self.state_manager = state_manager
        self.router = router
        self.unrelated_handler = unrelated_handler
        self.repo = session_repo
        self.llm = llm

    async def process_task_stream(self, user_input: str, session_id: str = None) -> AsyncGenerator[str, None]:
        sid = session_id or ""
        lock = load_lock(self.repo, sid)

        # --- Gate: reuse locked intent for continuation turns. ---
        if lock.is_active and self._is_continuation(user_input, lock):
            audit_event(layer="orchestrator", event_type="context_lock_hit",
                        message="reuse locked intent={}".format(lock.intent),
                        data={"intent": lock.intent, "input": user_input[:50]})
            save_lock(self.repo, sid, lock.intent, lock.params)  # refresh TTL
            async for token in self.router.route(lock.intent, user_input, session_id=sid):
                yield token
            return

        # --- Switch keyword: clear lock, fall through to classify. ---
        if self._is_switch(user_input):
            clear_lock(self.repo, sid)

        raw = ""
        async for token in self.classifier.classify_stream(user_input):
            raw += token

        result = self._parse_json(raw)
        intent_type = result.get("intent_type", "knowledge_qa")
        topic = result.get("topic", "N/A")
        logger.info("Classified as: %s (topic: %s)", intent_type, topic)

        audit_event(layer="orchestrator", event_type="intent_decision",
                    message="intent={} topic={}".format(intent_type, topic),
                    data={"intent_type": intent_type, "topic": topic,
                          "required_fields": result.get("required_fields", {}),
                          "missing_fields": result.get("missing_fields", [])})

        # --- Auto-overwrite: new intent != locked intent -> clear old params. ---
        if lock.intent and intent_type != lock.intent:
            clear_lock(self.repo, sid)
        # Acquire / refresh lock for actionable intents.
        if intent_type in ("environment_match", "knowledge_qa"):
            save_lock(self.repo, sid, intent_type, result.get("required_fields", {}))

        if intent_type == "unrelated":
            reply = await self.unrelated_handler.handle(user_input)
            yield reply
        else:
            async for token in self.router.route(intent_type, user_input, session_id=sid):
                yield token

    # ---- continuation judgment (two-step, entry-point logic) ----
    def _is_switch(self, text: str) -> bool:
        t = text.strip()
        return any(w in t for w in SWITCH_WORDS)

    def _is_continuation(self, text: str, lock: ContextLock) -> bool:
        t = text.strip()
        if self._is_switch(t):
            return False
        # short sentence or reference word -> continuation
        if len(t) <= 8 or any(w in t for w in REFERENCE_WORDS):
            return True
        # same-domain keyword for env-match lock
        if lock.intent == "environment_match" and self._has_env_keyword(t):
            return True
        # semantically deviating -> lightweight LLM fallback
        return self._llm_judge(t, lock)

    def _has_env_keyword(self, text: str) -> bool:
        low = text.lower()
        return any(w in low for w in ENV_WORDS)

    def _llm_judge(self, text: str, lock: ContextLock) -> bool:
        if not self.llm:
            return False  # conservative: no llm -> re-classify
        try:
            from langchain_core.messages import HumanMessage
            tmpl = (_PROMPTS / "context_lock_judge_v1.txt").read_text(encoding="utf-8")
            prompt = tmpl.format(intent=lock.intent,
                                 params=json.dumps(lock.params, ensure_ascii=False),
                                 input=text)
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            return resp.content.strip().startswith("是")
        except Exception as e:
            logger.warning("lock LLM judge failed: %s", e)
            return False

    def _parse_json(self, text: str) -> dict:
        try:
            json_text = self._extract_json_object(text)
            if json_text:
                return json.loads(json_text)
        except Exception:
            pass
        return {"intent_type": "knowledge_qa", "required_fields": {},
                "missing_fields": [], "keywords": [], "topic": ""}

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        """Extract the outermost JSON object from text, supporting nested braces."""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def reset_conversation(self):
        self.state_manager.reset()
