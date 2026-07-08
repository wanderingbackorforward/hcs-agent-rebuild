"""Trace data model — the common data backbone for all 5 evaluation metrics.

Every metric (task success / answer quality / trajectory / tool call / latency-token)
reads from a single ``AgentTrace``. This decouples *collection* from *scoring*:

- **Live mode**: a ``TraceRecorder`` is attached to a running agent; it records
  each ReAct step, tool call, timing, and token usage as they happen.
- **Replay mode**: a pre-recorded or synthetic ``AgentTrace`` is fed straight into
  the metric calculators — no LLM/API key required. This is what CI uses.

Design goal: ``AgentTrace`` captures enough that all 5 metrics can be computed
from it alone, without re-running the agent.
"""
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolCallRecord:
    """One tool invocation inside a ReAct step."""

    step: int
    tool_name: str
    args: Dict[str, Any] = field(default_factory=dict)
    result: str = ""
    success: bool = True
    error: str = ""
    latency_ms: float = 0.0


@dataclass
class TraceStep:
    """One ReAct iteration: Thought -> Action -> Observation."""

    step: int
    thought: str = ""
    action: str = ""
    observation: str = ""
    tool_call: Optional[ToolCallRecord] = None
    is_final: bool = False
    latency_ms: float = 0.0


# Termination reasons — consumed by the trajectory metric.
REASON_FINAL_ANSWER = "final_answer"
REASON_MAX_ITERATIONS = "max_iterations"
REASON_NO_ACTION = "no_action"
REASON_ERROR = "error"
REASON_MANUAL = "manual"


@dataclass
class AgentTrace:
    """A full agent run: from query to final answer, with every step in between."""

    query: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    session_id: str = ""
    steps: List[TraceStep] = field(default_factory=list)
    final_answer: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    success: bool = False
    termination_reason: str = ""
    # Retrieved context used to ground the answer — needed for hallucination check.
    retrieved_context: str = ""
    # Free-form metadata (intent, agent_name, labels, ...).
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def latency_ms(self) -> float:
        end = self.ended_at or time.time()
        return max(0.0, (end - self.started_at) * 1000.0)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def tool_calls(self) -> List[ToolCallRecord]:
        return [s.tool_call for s in self.steps if s.tool_call is not None]

    @property
    def step_count(self) -> int:
        return len(self.steps)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "query": self.query,
            "final_answer": self.final_answer,
            "success": self.success,
            "termination_reason": self.termination_reason,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "step_count": self.step_count,
            "steps": [
                {
                    "step": s.step, "thought": s.thought, "action": s.action,
                    "observation": s.observation[:200], "is_final": s.is_final,
                    "tool_call": (
                        None if s.tool_call is None
                        else {
                            "tool": s.tool_call.tool_name,
                            "args": s.tool_call.args,
                            "success": s.tool_call.success,
                            "error": s.tool_call.error,
                        }
                    ),
                }
                for s in self.steps
            ],
            "meta": self.meta,
        }


class TraceRecorder:
    """Collects one ``AgentTrace`` per agent run.

    Attach to an agent (e.g. ``ReActLoop``) in live mode. In replay mode you
    build ``AgentTrace`` objects directly and skip the recorder entirely.
    """

    def __init__(self) -> None:
        self.traces: List[AgentTrace] = []
        self._current: Optional[AgentTrace] = None

    def start(self, query: str, session_id: str = "", trace_id: str = "") -> AgentTrace:
        trace = AgentTrace(query=query, session_id=session_id, trace_id=trace_id)
        self._current = trace
        return trace

    @property
    def current(self) -> Optional[AgentTrace]:
        return self._current

    def add_step(self, step: TraceStep) -> None:
        if self._current is not None:
            self._current.steps.append(step)

    def add_tool_call(self, tool_call: ToolCallRecord) -> None:
        if self._current is not None and self._current.steps:
            self._current.steps[-1].tool_call = tool_call

    def record_tokens(self, prompt_tokens: int, completion_tokens: int) -> None:
        if self._current is not None:
            self._current.prompt_tokens += prompt_tokens
            self._current.completion_tokens += completion_tokens

    def set_retrieved_context(self, context: str) -> None:
        if self._current is not None:
            self._current.retrieved_context = context

    def finalize(
        self,
        final_answer: str,
        success: bool,
        termination_reason: str,
    ) -> Optional[AgentTrace]:
        if self._current is None:
            return None
        self._current.final_answer = final_answer
        self._current.success = success
        self._current.termination_reason = termination_reason
        self._current.ended_at = time.time()
        self.traces.append(self._current)
        done = self._current
        self._current = None
        return done
