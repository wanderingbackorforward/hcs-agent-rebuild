"""Tests for the 5-metric Agent evaluation framework.

All tests use replay mode (synthetic traces) or a FakeLLM — no real API key
required, so they run in CI. Covers: trace model, all 5 metric calculators,
offline evaluator, online evaluator, and ReAct-loop instrumentation.
"""
import asyncio
import pytest

from eval.trace import (
    AgentTrace, TraceStep, ToolCallRecord, TraceRecorder,
    REASON_FINAL_ANSWER, REASON_MAX_ITERATIONS, REASON_NO_ACTION,
)
from eval.metrics_content import (
    task_success_rate, answer_quality, hallucination_score,
)
from eval.metrics_process import trajectory_quality, tool_call_quality
from eval.metrics_cost import aggregate_cost
from eval.metrics import compute_all_metrics, compute_single_trace_metrics
from eval.golden_cases import get_cases, get_replay_traces, TOOL_SCHEMAS
from eval.offline import OfflineEvaluator
from eval.online import OnlineEvaluator


def _trace(answer="", success=True, reason=REASON_FINAL_ANSWER, **kw):
    t = AgentTrace(query=kw.get("query", "q"), final_answer=answer,
                   success=success, termination_reason=reason)
    t.prompt_tokens = kw.get("pt", 100)
    t.completion_tokens = kw.get("ct", 50)
    t.retrieved_context = kw.get("ctx", "")
    t.ended_at = t.started_at + kw.get("dur", 1.0)
    return t


# --- Trace model ---
class TestTraceModel:
    def test_tokens_and_latency(self):
        t = _trace(pt=100, ct=50, dur=1.0)
        assert t.total_tokens == 150
        assert t.latency_ms >= 900

    def test_tool_calls_property(self):
        t = AgentTrace(query="q")
        t.steps.append(TraceStep(step=1,
            tool_call=ToolCallRecord(step=1, tool_name="t")))
        assert len(t.tool_calls) == 1

    def test_recorder(self):
        rec = TraceRecorder()
        rec.start("q")
        rec.add_step(TraceStep(step=1))
        done = rec.finalize("ans", True, REASON_FINAL_ANSWER)
        assert done.success and len(rec.traces) == 1


# --- 1. 任务成功率 ---
class TestTaskSuccess:
    def test_success(self):
        t = _trace(answer="HCS hybrid cloud")
        r = task_success_rate([t], [{"expected_keywords": ["hybrid"]}])
        assert r["task_success_rate"] == 1.0

    def test_no_keyword(self):
        t = _trace(answer="nope")
        assert task_success_rate([t], [{"expected_keywords": ["hybrid"]}])["task_success_rate"] == 0.0


# --- 2. 回答质量 + 幻觉 ---
class TestAnswerQuality:
    def test_grounded(self):
        t = _trace(answer="HCS 混合云", ctx="HCS 混合云")
        q = answer_quality(t, "HCS 混合云")
        assert q["hallucination_safety"] >= 0.8

    def test_hallucination(self):
        t = _trace(answer="部署在火星数据中心", ctx="部署在本地机房")
        assert hallucination_score(t.final_answer, t.retrieved_context) < 0.5


# --- 3. 执行轨迹 ---
class TestTrajectory:
    def test_clean(self):
        t = _trace()
        t.steps.append(TraceStep(step=1, is_final=True))
        r = trajectory_quality(t)
        assert r["trajectory_score"] == 1.0 and r["terminated_cleanly"]

    def test_dead_loop(self):
        t = _trace(success=False, reason=REASON_MAX_ITERATIONS)
        for i in (1, 2):
            t.steps.append(TraceStep(step=i,
                tool_call=ToolCallRecord(step=i, tool_name="t", args={"q": "x"})))
        r = trajectory_quality(t)
        assert "dead_loop" in str(r["issues"]) and r["trajectory_score"] < 1.0

    def test_premature_stop(self):
        t = _trace(success=False, reason=REASON_NO_ACTION)
        t.steps.append(TraceStep(step=1))
        assert trajectory_quality(t)["trajectory_score"] < 1.0


# --- 4. 工具调用 ---
class TestToolCall:
    def test_correct(self):
        t = AgentTrace(query="q")
        t.steps.append(TraceStep(step=1, tool_call=ToolCallRecord(
            step=1, tool_name="query_knowledge_hub",
            args={"query": "HCS"}, success=True)))
        r = tool_call_quality(t, ["query_knowledge_hub"], TOOL_SCHEMAS)
        assert r["selection_accuracy"] == 1.0 and r["param_validity"] == 1.0

    def test_missing_param(self):
        t = AgentTrace(query="q")
        t.steps.append(TraceStep(step=1, tool_call=ToolCallRecord(
            step=1, tool_name="get_document_summary", args={}, success=False)))
        assert tool_call_quality(t, None, TOOL_SCHEMAS)["param_validity"] == 0.0

    def test_wrong_tool(self):
        t = AgentTrace(query="q")
        t.steps.append(TraceStep(step=1, tool_call=ToolCallRecord(
            step=1, tool_name="get_document_summary",
            args={"doc_id": "x"}, success=True)))
        assert tool_call_quality(t, ["query_knowledge_hub"], TOOL_SCHEMAS)["selection_accuracy"] == 0.0


# --- 5. 延迟和Token ---
class TestCost:
    def test_percentiles(self):
        traces = []
        for ms in [100, 200, 300, 400, 5000]:
            t = AgentTrace(query="q", prompt_tokens=100, completion_tokens=50)
            t.started_at, t.ended_at, t.success = 0, ms / 1000.0, True
            traces.append(t)
        c = aggregate_cost(traces)
        assert c["p50_latency_ms"] >= 100
        assert c["p95_latency_ms"] >= c["p50_latency_ms"]
        assert c["avg_total_tokens"] == 150


# --- Offline evaluator ---
class TestOffline:
    def test_replay_run(self):
        report = OfflineEvaluator(mode="replay").run()
        assert report["case_count"] == 6
        assert "overall_score" in report and "markdown" in report
        assert report["1_task_success_rate"]["task_success_rate"] > 0

    def test_compute_all(self):
        traces = get_replay_traces()
        cases = get_cases()
        report = compute_all_metrics(
            traces,
            [{"expected_keywords": c["expected_keywords"], "require_final": c["require_final"]} for c in cases],
            [c["reference_answer"] for c in cases],
            [c["expected_tools"] for c in cases], TOOL_SCHEMAS)
        assert 0 <= report["overall_score"] <= 1


# --- Online evaluator ---
class TestOnline:
    def test_snapshot(self):
        ev = OnlineEvaluator(window_size=10)
        assert ev.snapshot()["status"] == "no_data"
        t = _trace(answer="a")
        ev.record_trace(t)
        snap = ev.snapshot()
        assert snap["status"] == "ok" and snap["sampled"] == 1

    def test_recent(self):
        ev = OnlineEvaluator()
        ev.record_trace(_trace(answer="a1"))
        assert len(ev.recent_traces(5)) == 1


# --- ReAct-loop instrumentation (FakeLLM) ---
class TestReActInstrumentation:
    def test_captures_trace(self):
        from agents.knowledge_qa.react_loop import ReActLoop

        class FakeLLM:
            def __init__(self):
                self._n = 0

            async def ainvoke(self, msgs):
                from langchain_core.messages import AIMessage
                self._n += 1
                if self._n == 1:
                    return AIMessage(content='Thought: 查一下\nAction: {"tool":"echo","args":{"q":"HCS"}}')
                return AIMessage(content="Thought: 够了\nFinal Answer: HCS 是混合云")

        def echo(q):
            return "HCS 是混合云"

        rec = TraceRecorder()
        loop = ReActLoop(FakeLLM(), {"echo": echo}, recorder=rec)
        ans = asyncio.run(loop.run("HCS 是什么？"))
        assert "混合云" in ans
        assert len(rec.traces) == 1
        tr = rec.traces[0]
        assert tr.success and tr.termination_reason == "final_answer"
        assert tr.step_count == 2
        assert tr.tool_calls[0].tool_name == "echo"
