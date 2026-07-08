"""Offline evaluator — runs the golden set and scores all 5 metrics.

Two modes:

- **Replay mode** (default, no LLM key needed): scores pre-recorded synthetic
  traces from ``golden_cases.get_replay_traces()``. Perfect for CI.
- **Live mode**: runs each golden query through the real agent with a
  ``TraceRecorder`` attached, then scores the captured traces. Requires LLM +
  Embedding API keys; auto-selected when keys are present.

Usage::

    from eval.offline import OfflineEvaluator

    ev = OfflineEvaluator(mode="auto")
    report = ev.run()
    print(report.to_markdown())
"""
import logging
import os
from typing import List, Optional

from eval.trace import AgentTrace, TraceRecorder
from eval.golden_cases import (
    get_cases, get_replay_traces, TOOL_SCHEMAS,
)
from eval.metrics import compute_all_metrics
from eval.report import render_report

logger = logging.getLogger(__name__)


def _has_llm_key() -> bool:
    return bool(os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"))


class OfflineEvaluator:
    """Run the golden set and produce a scored report."""

    def __init__(self, mode: str = "auto", cases: Optional[List[dict]] = None):
        """
        Args:
            mode: "auto" | "live" | "replay". "auto" picks live when an LLM
                key is present, else replay.
            cases: override golden cases (mainly for tests).
        """
        if mode == "auto":
            mode = "live" if _has_llm_key() else "replay"
        self.mode = mode
        self.cases = cases or get_cases()

    def run(self) -> dict:
        """Execute the golden set and return the full metrics report dict."""
        logger.info("Offline eval running in %s mode (%d cases)",
                    self.mode, len(self.cases))
        if self.mode == "replay":
            traces = self._run_replay()
        else:
            traces = self._run_live()
        return self._score(traces)

    def _run_replay(self) -> List[AgentTrace]:
        traces = get_replay_traces()
        # Align length with cases (defensive).
        if len(traces) < len(self.cases):
            traces = traces + [traces[-1]] * (len(self.cases) - len(traces))
        return traces[: len(self.cases)]

    def _run_live(self) -> List[AgentTrace]:
        """Run each golden query through the real agent, capturing a trace."""
        traces: List[AgentTrace] = []
        recorder = TraceRecorder()
        try:
            from api.chat_handler import process_user_input
        except Exception as e:  # import / init failure → fall back to replay
            logger.warning("Live agent unavailable (%s); falling back to replay.", e)
            self.mode = "replay"
            return self._run_replay()

        for case in self.cases:
            recorder.start(query=case["query"])
            try:
                answer = process_user_input(case["query"])
                recorder.finalize(
                    final_answer=answer, success=True,
                    termination_reason="final_answer",
                )
            except Exception as e:
                logger.warning("Case %s failed: %s", case.get("id"), e)
                recorder.finalize(
                    final_answer="", success=False,
                    termination_reason="error",
                )
            if recorder.traces:
                traces.append(recorder.traces[-1])
        return traces

    def _score(self, traces: List[AgentTrace]) -> dict:
        cases = self.cases
        expected = [
            {"expected_keywords": c.get("expected_keywords", []),
             "require_final": c.get("require_final", True)}
            for c in cases
        ]
        references = [c.get("reference_answer", "") for c in cases]
        expected_tools = [c.get("expected_tools") for c in cases]
        report = compute_all_metrics(
            traces, expected=expected, references=references,
            expected_tools_per_case=expected_tools, tool_schemas=TOOL_SCHEMAS,
        )
        report["mode"] = self.mode
        report["markdown"] = render_report(report, traces, cases)
        report["traces"] = traces
        return report
