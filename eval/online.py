"""Online evaluator — samples production traffic and streams the 5 metrics.

Two parts:

1. ``OnlineEvaluator`` — a rolling window of recent traces + a live snapshot
   of all 5 metrics. Thread-safe, meant to be a singleton attached to the app.
2. ``EvalMiddleware`` — a FastAPI middleware that samples ``/chat`` requests,
   times them, builds a (partial) ``AgentTrace``, and feeds the evaluator.

In production the agent's ``TraceRecorder`` (see ``react_loop.py``) can publish
its full trace into the evaluator via ``record_trace`` so trajectory + tool-call
metrics are populated online too. Without that, the middleware still captures
query / answer / latency / success — enough for metrics 1, 2, 5.

Also collects negative samples (user rejections + NLI rejects) for periodic
Few-Shot / fine-tuning updates. Exposes ``GET /eval/metrics``,
``GET /eval/traces``, and ``GET /eval/negative-samples`` on the app.
"""
import logging
import random
import time
import threading
from typing import Any, Dict, List, Optional

from eval.trace import AgentTrace, REASON_FINAL_ANSWER, REASON_ERROR
from eval.metrics import compute_all_metrics

logger = logging.getLogger(__name__)

_DEFAULT_WINDOW = 200
_NEG_SAMPLE_LIMIT = 1000  # cap stored negative samples


class OnlineEvaluator:
    """Rolling-window store + live metric snapshot for production traffic."""

    def __init__(self, window_size: int = _DEFAULT_WINDOW):
        self.window_size = window_size
        self._traces: List[AgentTrace] = []
        self._negative_samples: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._sample_count = 0
        self._total_count = 0
        self._offline_accuracy: Optional[float] = None  # set from golden eval

    def record_trace(self, trace: AgentTrace) -> None:
        with self._lock:
            self._traces.append(trace)
            if len(self._traces) > self.window_size:
                self._traces = self._traces[-self.window_size:]
            self._sample_count += 1

    def record_negative_sample(
        self, query: str, routed_intent: str, reason: str,
        session_id: str = "",
    ) -> None:
        """Record a negative sample for periodic Few-Shot / fine-tuning.

        Sources:
          - user_rejection: user explicitly denies the routing result
          - nli_reject: NLI validator caught a routing mismatch
        """
        with self._lock:
            self._negative_samples.append({
                "query": query[:200],
                "routed_intent": routed_intent,
                "reason": reason,
                "session_id": session_id,
                "timestamp": time.time(),
            })
            if len(self._negative_samples) > _NEG_SAMPLE_LIMIT:
                self._negative_samples = self._negative_samples[-_NEG_SAMPLE_LIMIT:]

    def record_request(self, total: bool = True) -> None:
        with self._lock:
            if total:
                self._total_count += 1

    def set_offline_accuracy(self, accuracy: float) -> None:
        """Set the latest offline golden-set accuracy for comparison."""
        with self._lock:
            self._offline_accuracy = accuracy

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            traces = list(self._traces)
            neg_samples = list(self._negative_samples)
            offline_acc = self._offline_accuracy
        if not traces:
            return {"status": "no_data", "sampled": 0,
                    "window_size": self.window_size,
                    "negative_samples": len(neg_samples)}
        # Online has no golden labels → score without expected/references.
        metrics = compute_all_metrics(traces)
        metrics["status"] = "ok"
        metrics["sampled"] = self._sample_count
        metrics["window_size"] = self.window_size
        metrics["negative_samples"] = len(neg_samples)
        # Online vs offline comparison.
        if offline_acc is not None:
            online_acc = metrics.get("overall_score", 0)
            metrics["offline_accuracy"] = offline_acc
            metrics["online_accuracy"] = online_acc
            metrics["gap"] = round(offline_acc - online_acc, 3)
        return metrics

    def recent_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            return [t.to_dict() for t in self._traces[-limit:]]

    def recent_negative_samples(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._negative_samples[-limit:])


# Module-level singleton (attached to app.state in main).
_evaluator: Optional[OnlineEvaluator] = None


def get_evaluator() -> OnlineEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = OnlineEvaluator()
    return _evaluator


def attach_to_app(app, sample_rate: float = 0.1) -> None:
    """Attach the online evaluator + eval routes to a FastAPI app.

    Args:
        app: FastAPI instance.
        sample_rate: fraction of /chat requests to sample (0..1).
    """
    ev = get_evaluator()
    ev._sample_rate = sample_rate  # type: ignore[attr-defined]

    @app.middleware("http")
    async def _eval_middleware(request, call_next):
        ev.record_request()
        if not _should_sample(request, sample_rate):
            return await call_next(request)
        return await _sampled_call(request, call_next, ev)

    @app.get("/eval/metrics")
    async def _metrics_endpoint():
        return ev.snapshot()

    @app.get("/eval/traces")
    async def _traces_endpoint(limit: int = 20):
        return {"traces": ev.recent_traces(limit)}

    @app.get("/eval/negative-samples")
    async def _neg_samples_endpoint(limit: int = 50):
        with ev._lock:
            samples = list(ev._negative_samples[-limit:])
        return {"samples": samples, "total": len(ev._negative_samples)}


def _should_sample(request, rate: float) -> bool:
    path = getattr(request.url, "path", "")
    if "/chat" not in path:
        return False
    if rate >= 1.0:
        return True
    return random.random() < rate


async def _sampled_call(request, call_next, ev: OnlineEvaluator):
    start = time.time()
    query = ""
    try:
        body = await request.json()
        query = body.get("message", body.get("query", ""))
    except Exception:
        pass
    trace = AgentTrace(query=query, started_at=start)
    try:
        response = await call_next(request)
        ok = response.status_code < 400
    except Exception as e:
        logger.warning("online sample failed: %s", e)
        trace.success = False
        trace.termination_reason = REASON_ERROR
        trace.ended_at = time.time()
        ev.record_trace(trace)
        raise
    trace.success = ok
    trace.termination_reason = REASON_FINAL_ANSWER if ok else REASON_ERROR
    trace.ended_at = time.time()
    ev.record_trace(trace)
    return response
