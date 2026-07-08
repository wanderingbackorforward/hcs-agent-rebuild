"""Process metrics — 执行轨迹 (trajectory) & 工具调用 (tool call).

Both metrics look at the *process* / *capability*, not the final output:

- **执行轨迹**: did the agent loop, hit a dead end (max iterations), or terminate
  prematurely? Is the step count reasonable? Detects "乱调用 / 死循环 / 乱结束".
- **工具调用**: did it pick the right tool, with correct params (schema-valid),
  low error rate, and no redundant calls?
"""
from typing import Any, Dict, List, Optional

from eval.trace import (
    AgentTrace, REASON_FINAL_ANSWER, REASON_MAX_ITERATIONS,
    REASON_NO_ACTION, REASON_ERROR,
)


# --------------------------------------------------------------------------- #
# 3. 执行轨迹 (Execution Trajectory) — 看过程
# --------------------------------------------------------------------------- #

def trajectory_quality(trace: AgentTrace) -> Dict[str, Any]:
    """Score how clean the execution path was.

    Detects three failure modes and scores 0..1 (1.0 = ideal):
    - **dead_loop**: repeated identical tool calls (same tool + same args).
    - **dead_end**: hit max iterations without a Final Answer.
    - **premature_stop**: ended via no_action / error instead of final_answer.
    Also reports ``step_count`` and an efficiency flag.
    """
    score = 1.0
    reasons: List[str] = []

    # Dead-loop detection.
    seen: Dict[str, int] = {}
    loops = 0
    for tc in trace.tool_calls:
        key = "{}::{}".format(tc.tool_name, _stable_args(tc.args))
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            loops += 1
    if loops > 0:
        penalty = min(0.5, 0.25 * loops)
        score -= penalty
        reasons.append("dead_loop({}x)".format(loops))

    # Dead-end: max iterations without a final answer.
    if trace.termination_reason == REASON_MAX_ITERATIONS:
        score -= 0.4
        reasons.append("max_iterations_dead_end")

    # Premature termination: no_action or error.
    if trace.termination_reason in (REASON_NO_ACTION, REASON_ERROR):
        score -= 0.3
        reasons.append("premature_stop({})".format(trace.termination_reason))

    # Efficiency: too many steps for what should be simple.
    n = trace.step_count
    if n > 5:
        score -= 0.1
        reasons.append("too_many_steps({})".format(n))

    # Empty trajectory is suspicious.
    if n == 0:
        score -= 0.3
        reasons.append("empty_trajectory")

    score = max(0.0, min(1.0, score))
    return {
        "trajectory_score": round(score, 3),
        "step_count": n,
        "loop_count": loops,
        "terminated_cleanly": trace.termination_reason == REASON_FINAL_ANSWER,
        "termination_reason": trace.termination_reason,
        "issues": reasons,
    }


def aggregate_trajectory(traces: List[AgentTrace]) -> Dict[str, Any]:
    """Average trajectory quality across a batch."""
    if not traces:
        return {"trajectory_score": 0.0, "avg_steps": 0.0,
                "loop_rate": 0.0, "clean_termination_rate": 0.0}
    scores = [trajectory_quality(t) for t in traces]
    n = len(traces)
    return {
        "trajectory_score": round(sum(s["trajectory_score"] for s in scores) / n, 3),
        "avg_steps": round(sum(s["step_count"] for s in scores) / n, 3),
        "loop_rate": round(sum(1 for s in scores if s["loop_count"] > 0) / n, 3),
        "clean_termination_rate": round(
            sum(1 for s in scores if s["terminated_cleanly"]) / n, 3),
    }


def _stable_args(args: Dict[str, Any]) -> str:
    """Deterministic string key for loop detection (ignores arg order)."""
    try:
        return ",".join("{}={}".format(k, args.get(k)) for k in sorted(args))
    except Exception:
        return str(args)


# --------------------------------------------------------------------------- #
# 4. 工具调用 (Tool Call) — 看能力
# --------------------------------------------------------------------------- #

def tool_call_quality(
    trace: AgentTrace,
    expected_tools: Optional[List[str]] = None,
    tool_schemas: Optional[Dict[str, Dict]] = None,
) -> Dict[str, Any]:
    """Score tool selection + parameter correctness + error rate.

    Args:
        trace: the agent run.
        expected_tools: golden expectation — tools that *should* have been called.
        tool_schemas: ``{tool_name: {"required": [...], "properties": {...}}}``
            for parameter validation against the declared schema.
    """
    calls = trace.tool_calls
    if not calls:
        # No tool call may be fine for a trivial question, but if tools were
        # expected that is a selection failure.
        sel = 0.0 if expected_tools else 1.0
        return {"tool_score": round(sel, 3), "selection_accuracy": round(sel, 3),
                "param_validity": 1.0, "error_rate": 0.0, "call_count": 0,
                "redundant_calls": 0}

    called = [c.tool_name for c in calls]
    expected = expected_tools or []

    # Selection accuracy: fraction of expected tools actually called.
    if expected:
        hit = sum(1 for t in set(expected) if t in set(called))
        selection_accuracy = hit / len(set(expected))
    else:
        selection_accuracy = 1.0

    # Parameter validity: each call's required args present + types plausible.
    schema = tool_schemas or {}
    valid = 0
    for c in calls:
        if _validate_params(c, schema.get(c.tool_name)):
            valid += 1
    param_validity = valid / len(calls)

    # Error rate.
    errors = sum(1 for c in calls if not c.success)
    error_rate = errors / len(calls)

    # Redundant calls (same tool+args repeated).
    keys = ["{}::{}".format(c.tool_name, _stable_args(c.args)) for c in calls]
    redundant = sum(max(0, keys.count(k) - 1) for k in set(keys))

    # Combined score: weight selection 40%, params 30%, (1-error) 20%, no-redundancy 10%.
    redundancy_ratio = redundant / len(calls) if calls else 0.0
    score = (
        0.4 * selection_accuracy
        + 0.3 * param_validity
        + 0.2 * (1 - error_rate)
        + 0.1 * (1 - redundancy_ratio)
    )
    return {
        "tool_score": round(score, 3),
        "selection_accuracy": round(selection_accuracy, 3),
        "param_validity": round(param_validity, 3),
        "error_rate": round(error_rate, 3),
        "call_count": len(calls),
        "redundant_calls": redundant,
    }


def _validate_params(call, schema: Optional[Dict]) -> bool:
    """Check a tool call's args against its JSON-schema-ish declaration."""
    if not schema:
        return True  # no schema to check → assume valid
    required = schema.get("required", [])
    for req in required:
        if req not in call.args or call.args[req] in (None, ""):
            return False
    properties = schema.get("properties", {})
    for k, v in call.args.items():
        if k in properties:
            decl_type = properties[k].get("type")
            if decl_type and not _type_ok(v, decl_type):
                return False
    return True


def _type_ok(value: Any, decl_type: str) -> bool:
    types = {"string": str, "integer": int, "number": (int, float),
             "boolean": bool, "array": list, "object": dict}
    py = types.get(decl_type)
    if py is None:
        return True
    # bool is a subclass of int — exclude that false positive.
    if decl_type == "integer" and isinstance(value, bool):
        return False
    return isinstance(value, py)
