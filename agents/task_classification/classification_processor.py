"""Classification processor - orchestrates classify -> route -> respond pipeline.

Three-layer context-lock gate (reordered for safety):

  L1  switch-word fast intercept (0 cost) -> clear lock, full classify
  L2  confidence pre-check (multi-intent conflict / vague) -> clarify, don't execute
  L3  continuation check (short/ref/same-domain, else LLM judge)
      -> reuse lock, skip re-classification

Confidence is checked BEFORE continuation so that ambiguous inputs are
filtered out before L3 makes a routing decision on them.  This prevents
L3 from misjudging a multi-intent or vague input as a simple continuation.
Judgment lives here (entry point, one place); persistence is in
agents.context_lock.
"""
import json
import logging
from pathlib import Path
from typing import AsyncGenerator

from config.audit import audit_event
from agents.context_lock import ContextLock, load_lock, save_lock, clear_lock

logger = logging.getLogger(__name__)

_PROMPTS = Path(__file__).parent.parent.parent / "prompts"
SWITCH_WORDS = ("换个话题", "换话题", "退出", "不查了", "问别的", "换个问题",
                "算了", "新建任务", "不查环境", "不查这个")
REFERENCE_WORDS = ("那个", "这个", "换成", "改成", "再加", "再来一个",
                   "上面那个", "刚才", "它的", "另一个", "换一个")
ENV_SIGNAL = ("环境", "组件", "节点", "hbase", "kafka", "mysql", "redis",
              "区域", "region", "测试环境", "可用", "状态", "staging", "dev",
              "探测", "端口", "匹配", "筛选")
KB_SIGNAL = ("怎么", "是什么", "文档", "安装", "初始化", "配置说明", "规范",
             "手册", "有哪些", "区别", "在哪里", "部署阶段", "许可证")
MULTI_INTENT_MARKERS = ("顺便", "同时", "另外", "还有", "以及", "再帮我", "再问")
CONFIDENCE_THRESHOLD = 0.5


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

        # L1: switch-word fast intercept (0 cost).
        if self._is_switch(user_input):
            clear_lock(self.repo, sid)
            async for t in self._full_classify_route(user_input, sid, lock):
                yield t
            return

        # L2: confidence pre-check — filter ambiguous inputs before L3.
        # Only relevant when a lock is active (no lock -> full classify anyway).
        if lock.is_active:
            score, issue, question = self._assess_confidence(user_input, lock)
            if score < CONFIDENCE_THRESHOLD:
                audit_event(layer="orchestrator", event_type="low_confidence_clarify",
                            message="issue={} score={}".format(issue, score),
                            data={"issue": issue, "score": score, "input": user_input[:50]})
                yield question
                return

        # L3: continuation check (only on inputs that passed L2).
        if lock.is_active and await self._is_continuation(user_input, lock):
            audit_event(layer="orchestrator", event_type="context_lock_hit",
                        message="reuse locked intent={}".format(lock.intent),
                        data={"intent": lock.intent, "confidence": score})
            save_lock(self.repo, sid, lock.intent, lock.params)  # refresh TTL
            async for t in self.router.route(lock.intent, user_input, session_id=sid):
                yield t
            return

        # L3 said "not continuing" with an active lock -> clear it.
        if lock.is_active:
            clear_lock(self.repo, sid)
        # Full classify + route.
        async for t in self._full_classify_route(user_input, sid, lock):
            yield t

    async def _full_classify_route(self, user_input, sid, lock):
        """Step 6: full LLM classification, update/overwrite lock, route."""
        raw = ""
        history = self._load_history(sid)
        async for token in self.classifier.classify_stream(user_input, history=history):
            raw += token
        result = self._parse_json(raw)
        intent_type = result.get("intent_type", "knowledge_qa")
        topic = result.get("topic", "N/A")
        logger.info("Classified as: %s (topic: %s)", intent_type, topic)
        audit_event(layer="orchestrator", event_type="intent_decision",
                    message="intent={} topic={}".format(intent_type, topic),
                    data={"intent_type": intent_type, "topic": topic,
                          "required_fields": result.get("required_fields", {})})
        # Auto-overwrite: new intent != locked intent -> clear old params.
        if lock.intent and intent_type != lock.intent:
            clear_lock(self.repo, sid)
        if intent_type in ("environment_match", "knowledge_qa"):
            save_lock(self.repo, sid, intent_type, result.get("required_fields", {}))
        if intent_type == "unrelated":
            yield await self.unrelated_handler.handle(user_input)
        else:
            async for t in self.router.route(intent_type, user_input, session_id=sid):
                yield t

    # ---- L1 / L3: switch + continuation ----
    def _is_switch(self, text: str) -> bool:
        return any(w in text.strip() for w in SWITCH_WORDS)

    async def _is_continuation(self, text: str, lock: ContextLock) -> bool:
        t = text.strip()
        if self._is_switch(t):
            return False
        if len(t) <= 8 or any(w in t for w in REFERENCE_WORDS):
            return True
        # same-domain signal for the locked intent
        if lock.intent == "environment_match" and self._has_signal(t, ENV_SIGNAL):
            return True
        if lock.intent == "knowledge_qa" and self._has_signal(t, KB_SIGNAL):
            return True
        return await self._llm_judge(t, lock)

    # ---- L2: confidence pre-check (0 cost, no extra LLM) ----
    def _assess_confidence(self, text: str, lock: ContextLock):
        """Return (score, issue, clarify_question). Low score -> ask, don't execute."""
        t = text.strip()
        has_env = self._has_signal(t, ENV_SIGNAL)
        has_kb = self._has_signal(t, KB_SIGNAL)
        has_multi = any(w in t for w in MULTI_INTENT_MARKERS)
        # One sentence spanning two independent domains -> multi-intent conflict.
        if has_env and has_kb and has_multi:
            return 0.2, "multi_intent", (
                "你这句话同时涉及「环境匹配」和「技术问答」两个需求。"
                "想先处理哪一个——查/匹配环境，还是问技术问题？")
        # Vague pure-reference with no params to ground on.
        if len(t) <= 4 and self._is_pure_reference(t) and not lock.params:
            return 0.3, "vague", (
                "我没太明白你的意思，请补充具体的环境类型、组件或问题细节。")
        return 0.9, None, None

    @staticmethod
    def _has_signal(text: str, words) -> bool:
        low = text.lower()
        return any(w in low for w in words)

    @staticmethod
    def _is_pure_reference(text: str) -> bool:
        refs = ("那个", "这个", "换一个", "再来一个")
        return any(text == w or text.startswith(w) for w in refs)

    async def _llm_judge(self, text: str, lock: ContextLock) -> bool:
        if not self.llm:
            return False  # conservative: no llm -> re-classify
        try:
            from langchain_core.messages import HumanMessage
            tmpl = (_PROMPTS / "context_lock_judge_v1.txt").read_text(encoding="utf-8")
            prompt = tmpl.format(intent=lock.intent,
                                 params=json.dumps(lock.params, ensure_ascii=False),
                                 input=text)
            resp = await self.llm.ainvoke([HumanMessage(content=prompt)])
            return resp.content.strip().startswith("是")
        except Exception as e:
            logger.warning("lock LLM judge failed: %s", e)
            return False

    def _load_history(self, sid: str) -> list:
        """Load recent chat history from session repo for classifier context."""
        if not self.repo or not sid:
            return []
        try:
            return self.repo.get_history(sid)
        except Exception:
            return []

    def _parse_json(self, text: str) -> dict:
        try:
            j = self._extract_json_object(text)
            if j:
                return json.loads(j)
        except Exception:
            pass
        return {"intent_type": "knowledge_qa", "required_fields": {},
                "missing_fields": [], "keywords": [], "topic": ""}

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
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
