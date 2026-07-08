"""Cost metrics — 延迟 (latency) & Token (cost).

Looks at engineering viability: how fast, how expensive.

- **延迟**: per-request latency, plus P50 / P95 across a batch.
- **Token**: prompt / completion / total tokens per request, plus cost-per-
  successful-task (efficiency — wasted tokens on failed runs don't count).
"""
from typing import Dict, List

from eval.trace import AgentTrace


def latency_token_metrics(trace: AgentTrace) -> Dict[str, float]:
    """Single-trace latency + token report."""
    return {
        "latency_ms": round(trace.latency_ms, 1),
        "prompt_tokens": trace.prompt_tokens,
        "completion_tokens": trace.completion_tokens,
        "total_tokens": trace.total_tokens,
    }


def aggregate_cost(traces: List[AgentTrace]) -> Dict[str, float]:
    """Batch latency percentiles + token efficiency across traces."""
    if not traces:
        return {"p50_latency_ms": 0.0, "p95_latency_ms": 0.0,
                "avg_total_tokens": 0.0, "tokens_per_success": 0.0,
                "success_rate": 0.0}
    latencies = sorted(t.latency_ms for t in traces)
    totals = [t.total_tokens for t in traces]
    n = len(traces)
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)
    avg_tokens = sum(totals) / n
    successes = sum(1 for t in traces if t.success)
    success_tokens = sum(t.total_tokens for t in traces if t.success)
    tokens_per_success = success_tokens / successes if successes else float(totals[0] or 0)
    return {
        "p50_latency_ms": round(p50, 1),
        "p95_latency_ms": round(p95, 1),
        "avg_total_tokens": round(avg_tokens, 1),
        "tokens_per_success": round(tokens_per_success, 1),
        "success_rate": round(successes / n, 3),
    }


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Linear-interpolation percentile on an already-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = (len(sorted_values) - 1) * (pct / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac
