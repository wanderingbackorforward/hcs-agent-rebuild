"""Content metrics — 任务成功率 (task success) & 回答质量 (answer quality).

Both metrics look at the *outcome* / *output*, not the process:

- **任务成功率**: did the agent finish AND produce an answer matching the
  expected outcome? (keyword/label match; optional LLM judge)
- **回答质量**: accuracy (overlap with reference / LLM judge), completeness,
  conciseness, plus a **hallucination** signal — does the answer make claims
  unsupported by the retrieved context?
"""
from typing import Dict, List, Optional

from eval.trace import AgentTrace


# --------------------------------------------------------------------------- #
# 1. 任务成功率 (Task Success Rate) — 看结果
# --------------------------------------------------------------------------- #

def task_success_rate(traces: List[AgentTrace], expected: List[Dict]) -> Dict[str, float]:
    """Compute task success rate over a golden set.

    Args:
        traces: one trace per case (aligned with ``expected`` by index).
        expected: each ``{"expected_keywords": [...], "require_final": bool}``.

    Returns:
        ``{"task_success_rate": 0..1, "finished_rate": 0..1, "matched_rate": 0..1}``.
    """
    if not traces:
        return {"task_success_rate": 0.0, "finished_rate": 0.0, "matched_rate": 0.0}
    finished = matched = 0
    for tr, exp in zip(traces, expected):
        require_final = exp.get("require_final", True)
        kws = exp.get("expected_keywords", [])
        is_finished = (tr.termination_reason == "final_answer") or not require_final
        if is_finished:
            finished += 1
        ans = (tr.final_answer or "").lower()
        ok_match = tr.success and any(k.lower() in ans for k in kws) if kws else tr.success
        if ok_match:
            matched += 1
    n = len(traces)
    return {
        "task_success_rate": round(matched / n, 3),
        "finished_rate": round(finished / n, 3),
        "matched_rate": round(matched / n, 3),
    }


# --------------------------------------------------------------------------- #
# 2. 回答质量 (Answer Quality) — 看输出
# --------------------------------------------------------------------------- #

def _word_set(text: str) -> set:
    return set(text.lower().split())


def answer_quality(trace: AgentTrace, reference: str = "") -> Dict[str, float]:
    """Score a single answer: accuracy / completeness / conciseness / hallucination.

    Accuracy & completeness use heuristic overlap with ``reference`` when given.
    Hallucination checks whether the answer introduces entities absent from the
    retrieved context (the RAG grounding). Returns scores in 0..1; for
    hallucination, **higher is better** (1.0 = no hallucination detected).
    """
    answer = trace.final_answer or ""
    ans_words = _word_set(answer)

    # Accuracy: overlap with reference answer.
    if reference:
        ref_words = _word_set(reference)
        overlap = len(ref_words & ans_words)
        accuracy = min(1.0, overlap / max(len(ref_words), 1))
    else:
        accuracy = 0.5  # unknown — neutral prior

    # Completeness: length relative to a sensible reference length.
    ref_len = max(len(reference), 40)
    ans_len = len(answer)
    if ans_len == 0:
        completeness = 0.0
    elif ans_len <= ref_len * 2:
        completeness = min(1.0, ans_len / ref_len)
    else:
        completeness = 0.7  # too long → penalize

    # Conciseness: penalize verbosity.
    if ans_len <= ref_len * 1.5:
        conciseness = 1.0
    elif ans_len <= ref_len * 3:
        conciseness = 0.7
    else:
        conciseness = 0.4

    halluc = hallucination_score(answer, trace.retrieved_context)

    return {
        "accuracy": round(accuracy, 3),
        "completeness": round(completeness, 3),
        "conciseness": round(conciseness, 3),
        "hallucination_safety": round(halluc, 3),
    }


def hallucination_score(answer: str, context: str) -> float:
    """Estimate how grounded the answer is against retrieved context.

    Returns a *safety* score in 0..1 (1.0 = fully grounded, 0.0 = likely
    fabricated). Heuristic: fraction of content-word tokens in the answer that
    also appear in the context. When there is no context to check against,
    returns a neutral 1.0 (can't prove hallucination).
    """
    if not context.strip():
        return 1.0
    if not answer.strip():
        return 1.0
    ctx_words = _word_set(context)
    ans_words = _word_set(answer)
    # Drop trivial tokens for a fairer ratio.
    stop = {"的", "是", "了", "在", "和", "与", "the", "a", "an", "is", "are", "to", "of"}
    content_words = [w for w in ans_words if w not in stop and len(w) > 1]
    if not content_words:
        return 1.0
    grounded = sum(1 for w in content_words if w in ctx_words)
    return round(grounded / len(content_words), 3)


def aggregate_answer_quality(
    traces: List[AgentTrace], references: Optional[List[str]] = None
) -> Dict[str, float]:
    """Average answer-quality sub-scores across a batch of traces."""
    if not traces:
        return {"accuracy": 0.0, "completeness": 0.0,
                "conciseness": 0.0, "hallucination_safety": 0.0}
    refs = references or [""] * len(traces)
    sums = {"accuracy": 0.0, "completeness": 0.0,
            "conciseness": 0.0, "hallucination_safety": 0.0}
    for tr, ref in zip(traces, refs):
        q = answer_quality(tr, ref)
        for k in sums:
            sums[k] += q[k]
    n = len(traces)
    return {k: round(v / n, 3) for k, v in sums.items()}
