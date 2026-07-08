"""Metric aggregator — compute all 5 metrics from a batch of traces.

Re-exports the per-metric calculators and provides ``compute_all_metrics``,
the single entry point the offline / online evaluators call.

The 5 metrics map directly to the evaluation template:

  1. 任务成功率  (task_success_rate)   — 看结果
  2. 回答质量    (aggregate_answer_quality) — 看输出
  3. 执行轨迹    (aggregate_trajectory) — 看过程
  4. 工具调用    (aggregate_tool_call) — 看能力
  5. 延迟和Token (aggregate_cost)      — 看工程落地
"""
from typing import Any, Dict, List, Optional

from eval.trace import AgentTrace
from eval.metrics_content import (
    task_success_rate, aggregate_answer_quality, answer_quality,
)
from eval.metrics_process import (
    trajectory_quality, aggregate_trajectory,
    tool_call_quality,
)
from eval.metrics_cost import latency_token_metrics, aggregate_cost


def compute_all_metrics(
    traces: List[AgentTrace],
    expected: Optional[List[Dict]] = None,
    references: Optional[List[str]] = None,
    expected_tools_per_case: Optional[List[List[str]]] = None,
    tool_schemas: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """Compute all 5 metric families for a batch of traces.

    Args:
        traces: aligned batch of agent traces (one per golden case).
        expected: per-case ``{"expected_keywords":[...], "require_final":bool}``.
        references: per-case reference answer (for accuracy).
        expected_tools_per_case: per-case list of expected tool names.
        tool_schemas: ``{tool_name: {"required":[...], "properties":{...}}}``.
    """
    n = len(traces)
    exp = expected or [{}] * n
    refs = references or [""] * n
    exp_tools = expected_tools_per_case or [None] * n

    success = task_success_rate(traces, exp)
    quality = aggregate_answer_quality(traces, refs)
    trajectory = aggregate_trajectory(traces)
    cost = aggregate_cost(traces)

    # Tool-call aggregation (per-case then averaged).
    tool_scores = [tool_call_quality(t, et, tool_schemas)
                   for t, et in zip(traces, exp_tools)]
    tool_agg = _avg_tool_scores(tool_scores)

    # Overall score: equal-weighted blend of the 5 heads.
    overall = (
        0.25 * success["task_success_rate"]
        + 0.20 * _mean_quality(quality)
        + 0.20 * trajectory["trajectory_score"]
        + 0.20 * tool_agg["tool_score"]
        + 0.15 * _latency_norm(cost["p95_latency_ms"])
    )
    return {
        "overall_score": round(overall, 3),
        "1_task_success_rate": success,
        "2_answer_quality": quality,
        "3_execution_trajectory": trajectory,
        "4_tool_call": tool_agg,
        "5_latency_token": cost,
        "case_count": n,
    }


def _avg_tool_scores(scores: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not scores:
        return {"tool_score": 0.0, "selection_accuracy": 0.0,
                "param_validity": 0.0, "error_rate": 0.0,
                "avg_call_count": 0.0, "redundant_rate": 0.0}
    n = len(scores)
    keys = ["tool_score", "selection_accuracy", "param_validity", "error_rate"]
    out = {k: round(sum(s[k] for s in scores) / n, 3) for k in keys}
    out["avg_call_count"] = round(sum(s["call_count"] for s in scores) / n, 3)
    out["redundant_rate"] = round(
        sum(1 for s in scores if s["redundant_calls"] > 0) / n, 3)
    return out


def _mean_quality(quality: Dict[str, float]) -> float:
    return sum(quality.values()) / max(len(quality), 1)


def _latency_norm(p95_ms: float) -> float:
    """Normalize P95 latency to 0..1 (≤2s=1.0, ≥15s=0.0, linear between)."""
    if p95_ms <= 2000:
        return 1.0
    if p95_ms >= 15000:
        return 0.0
    return round(1.0 - (p95_ms - 2000) / 13000, 3)


# Per-trace single-shot metrics (used by online evaluator per request).
def compute_single_trace_metrics(
    trace: AgentTrace,
    expected: Optional[Dict] = None,
    reference: str = "",
    expected_tools: Optional[List[str]] = None,
    tool_schemas: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """Score one trace in isolation (online / per-request use)."""
    succ = task_success_rate([trace], [expected or {}])
    qual = answer_quality(trace, reference)
    traj = trajectory_quality(trace)
    tool = tool_call_quality(trace, expected_tools, tool_schemas)
    cost = latency_token_metrics(trace)
    return {
        "1_task_success_rate": succ,
        "2_answer_quality": qual,
        "3_execution_trajectory": traj,
        "4_tool_call": tool,
        "5_latency_token": cost,
    }
