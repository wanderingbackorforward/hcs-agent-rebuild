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
from config.constants import (
    SWITCH_WORDS, REFERENCE_WORDS, ENV_SIGNAL, KB_SIGNAL,
    MULTI_INTENT_MARKERS, CONFIDENCE_THRESHOLD,
)
from agents.context_lock import ContextLock, load_lock, save_lock, clear_lock
from agents.task_classification.json_utils import parse_classification_json
from agents.task_classification.semantic_checker import SemanticChecker
from agents.task_classification.nli_validator import NLIValidator, NLI_PASS_THRESHOLD, NLI_BORDERLINE_THRESHOLD
from config.settings import app_settings
from config.sse_protocol import SSEEvent
from config.decision_explainer import build_decision, agent_display_name

logger = logging.getLogger(__name__)

_PROMPTS = Path(__file__).parent.parent.parent / "prompts"


class ClassificationProcessor:
    def __init__(self, classifier, state_manager, router, unrelated_handler,
                 session_repo=None, llm=None, semantic_checker=None,
                 nli_validator=None):
        self.classifier = classifier
        self.state_manager = state_manager
        self.router = router
        self.unrelated_handler = unrelated_handler
        self.repo = session_repo
        self.llm = llm
        self.semantic_checker = semantic_checker or SemanticChecker()
        self.nli_validator = nli_validator or NLIValidator()

    async def process_task_stream(
        self, user_input: str, session_id: str = None,
        task_id: str = None,
    ) -> AsyncGenerator[str, None]:
        sid = session_id or ""
        lock = load_lock(self.repo, sid)

        # Cooperative cancellation check.
        if task_id and self._is_cancelled(task_id):
            return

        # L1: switch-word fast intercept (0 cost).
        if self._is_switch(user_input):
            clear_lock(self.repo, sid)
            yield SSEEvent.decision(**build_decision("switch"))
            yield SSEEvent.status("classifying", "检测到话题切换，重新理解需求...")
            async for t in self._full_classify_route(user_input, sid, lock, task_id):
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
                path = "clarify_multi" if issue == "multi_intent" else "clarify_vague"
                yield SSEEvent.decision(**build_decision(path, score=score))
                yield question
                return

        # L3: continuation check (only on inputs that passed L2).
        if lock.is_active and await self._is_continuation(user_input, lock):
            audit_event(layer="orchestrator", event_type="context_lock_hit",
                        message="reuse locked intent={}".format(lock.intent),
                        data={"intent": lock.intent, "confidence": score})
            save_lock(self.repo, sid, lock.intent, lock.params)  # refresh TTL
            yield SSEEvent.decision(**build_decision(
                "lock_hit", intent_type=lock.intent,
                context_lock_status="active",
            ))
            yield SSEEvent.status("continuing", "延续上轮对话，正在处理...")
            async for t in self.router.route(lock.intent, user_input, session_id=sid):
                yield t
            return

        # L3 said "not continuing" with an active lock -> clear it.
        if lock.is_active:
            clear_lock(self.repo, sid)
        # Full classify + route.
        async for t in self._full_classify_route(user_input, sid, lock, task_id):
            yield t

    async def _full_classify_route(self, user_input, sid, lock, task_id=None):
        """Full LLM classification, update/overwrite lock, route."""
        if task_id and self._is_cancelled(task_id):
            return
        yield SSEEvent.status("classifying", "正在理解您的需求...")
        raw = ""
        history = self._load_history(sid)
        async for token in self.classifier.classify_stream(user_input, history=history):
            raw += token
        result = parse_classification_json(raw)
        intent_type = result.get("intent_type", "knowledge_qa")
        topic = result.get("topic", "N/A")
        confidence = float(result.get("confidence", 1.0))
        logger.info("Classified as: %s (topic: %s, confidence: %.2f)", intent_type, topic, confidence)
        audit_event(layer="orchestrator", event_type="intent_decision",
                    message="intent={} topic={} confidence={:.2f}".format(intent_type, topic, confidence),
                    data={"intent_type": intent_type, "topic": topic, "confidence": confidence,
                          "required_fields": result.get("required_fields", {})})
        # Low-confidence classification -> ask for clarification instead of routing.
        if confidence < CONFIDENCE_THRESHOLD:
            audit_event(layer="orchestrator", event_type="low_confidence_clarify",
                        message="full-classify low confidence {:.2f}".format(confidence),
                        data={"intent_type": intent_type, "confidence": confidence,
                              "input": user_input[:50]})
            yield SSEEvent.decision(**build_decision(
                "low_confidence", intent_type=intent_type, confidence=confidence,
            ))
            yield "我不太确定您的需求。您是想查询/匹配测试环境，还是询问技术文档和规范方面的问题？请稍作补充。"
            return

        # ---- Step 4: NLI confidence gate (pluggable) ----
        # LLM has selected an agent. NLI independently checks whether the query
        # matches that agent's responsibility. NLI does NOT participate in
        # selection — it only provides an objective confidence score.
        nli_score = None
        if intent_type != "unrelated":
            nli_score = await self.nli_validator.nli_check(user_input, intent_type)

        if nli_score is not None:
            # NLI available — use objective score for routing decision.
            audit_event(layer="orchestrator", event_type="nli_check",
                        message="agent={} nli_score={:.3f} llm_conf={:.2f}".format(
                            intent_type, nli_score, confidence),
                        data={"intent_type": intent_type, "nli_score": nli_score,
                              "llm_confidence": confidence})
            if nli_score < NLI_BORDERLINE_THRESHOLD:
                # NLI says query doesn't match selected agent → fallback.
                yield SSEEvent.decision(**build_decision(
                    "nli_reject", intent_type=intent_type,
                    confidence=confidence, nli_score=nli_score,
                ))
                yield "抱歉，我无法准确匹配您的需求到对应的处理能力。请尝试更明确地描述您是想查询/匹配测试环境，还是询问技术文档和规范方面的问题。"
                return
            # Score in [borderline, pass) → route but flag as low-confidence.
            if nli_score < NLI_PASS_THRESHOLD:
                yield SSEEvent.decision(**build_decision(
                    "nli_borderline", intent_type=intent_type,
                    confidence=confidence, nli_score=nli_score,
                ))
            else:
                yield SSEEvent.decision(**build_decision(
                    "nli_pass", intent_type=intent_type,
                    confidence=confidence, nli_score=nli_score,
                ))
        else:
            # NLI unavailable (disabled or no embedder) → degradation path:
            # LLM self-confidence + keyword cross-check.
            keyword_hit = self._check_agent_keyword(user_input, intent_type)
            if confidence < app_settings.nli_fallback_confidence or not keyword_hit:
                audit_event(layer="orchestrator", event_type="nli_fallback_clarify",
                            message="degraded llm_conf={:.2f} kw_hit={}".format(
                                confidence, keyword_hit),
                            data={"intent_type": intent_type,
                                  "llm_confidence": confidence,
                                  "keyword_hit": keyword_hit})
                yield SSEEvent.decision(**build_decision(
                    "low_confidence", intent_type=intent_type, confidence=confidence,
                ))
                yield "我不太确定您的需求。您是想查询/匹配测试环境，还是询问技术文档和规范方面的问题？请稍作补充。"
                return
        # Emit classified decision event for user-facing explainability.
        yield SSEEvent.decision(**build_decision(
            "classified", intent_type=intent_type, confidence=confidence,
            agent=agent_display_name(intent_type),
        ))
        # Auto-overwrite: new intent != locked intent -> clear old params.
        if lock.intent and intent_type != lock.intent:
            clear_lock(self.repo, sid)
        if intent_type in ("environment_match", "knowledge_qa"):
            save_lock(self.repo, sid, intent_type, result.get("required_fields", {}))
        # Cancellation check + checkpoint before routing.
        if task_id and self._is_cancelled(task_id):
            self._save_checkpoint(task_id, stage="pre_route",
                                   intent_type=intent_type, sid=sid)
            return
        if intent_type == "unrelated":
            yield await self.unrelated_handler.handle(user_input)
        else:
            async for t in self.router.route(intent_type, user_input, session_id=sid):
                yield t

    # ---- Task cancellation helpers (lazy import to avoid circular) ----
    @staticmethod
    def _is_cancelled(task_id: str) -> bool:
        from api.task_manager import get_task_manager
        return get_task_manager().is_cancelled(task_id)

    @staticmethod
    def _save_checkpoint(task_id: str, **state) -> None:
        from api.task_manager import get_task_manager
        get_task_manager().checkpoint(task_id, state)

    # ---- L1 / L3: switch + continuation ----
    def _is_switch(self, text: str) -> bool:
        return any(w in text.strip() for w in SWITCH_WORDS)

    async def _is_continuation(self, text: str, lock: ContextLock) -> bool:
        t = text.strip()
        if self._is_switch(t):
            return False
        if len(t) <= 8 or any(w in t for w in REFERENCE_WORDS):
            return True
        # Semantic similarity check (embedding-based, replaces keyword matching).
        if self.semantic_checker and self.semantic_checker.is_available:
            result = await self.semantic_checker.check_continuation(t, lock)
            if result is not None:
                audit_event(layer="orchestrator", event_type="semantic_continuation",
                            message="sim_check intent={} result={}".format(lock.intent, result),
                            data={"intent": lock.intent, "semantic_result": result})
                return result
        # Fallback: same-domain signal for the locked intent
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
    def _check_agent_keyword(text: str, intent_type: str) -> bool:
        """Degradation helper: check if query contains keywords for the
        selected agent. Used when NLI is unavailable."""
        low = text.lower()
        if intent_type == "environment_match":
            return any(w in low for w in ENV_SIGNAL)
        if intent_type == "knowledge_qa":
            return any(w in low for w in KB_SIGNAL)
        return True  # unrelated doesn't need keyword check

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

    def reset_conversation(self):
        self.state_manager.reset()
