"""ReAct loop - Thought/Action/Observation cycle for KnowledgeQAAgent.

Interview talking point: "My KnowledgeQAAgent supports ReAct mode — LLM
outputs Thought and Action, executes the tool, injects Observation,
cycling up to 5 rounds."
"""
import json
import logging
import time
from typing import Dict, Optional
from langchain_core.messages import HumanMessage

from agents.context_manager import count_tokens
from config.settings import app_settings
from prompts.loader import load_prompt

logger = logging.getLogger(__name__)
MAX_ITERATIONS = app_settings.react_max_iterations

# Termination reasons (mirror eval.trace constants to avoid import cycle).
_REASON_FINAL = "final_answer"
_REASON_MAX_ITER = "max_iterations"
_REASON_NO_ACTION = "no_action"
_REASON_ERROR = "error"

_REACT_PROMPT_FILE = "react_v1.txt"


class ReActLoop:
    """ReAct loop: Thought -> Action -> Observation -> repeat."""

    def __init__(self, llm, tools: Dict, max_iterations: int = MAX_ITERATIONS,
                 recorder=None):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        # Optional eval trace recorder (eval.trace.TraceRecorder). When None,
        # behavior is identical to the un-instrumented loop.
        self.recorder = recorder

    async def run(self, question: str, conversation_context: str = "") -> str:
        scratchpad = ""
        if self.recorder:
            self.recorder.start(query=question)
        for i in range(self.max_iterations):
            step_start = time.time()
            prompt = load_prompt(_REACT_PROMPT_FILE).format(
                conversation_context=conversation_context or "(无历史)",
                scratchpad=scratchpad or "(无)",
                question=question,
            )

            try:
                result = await self.llm.ainvoke([HumanMessage(content=prompt)])
                response = result.content.strip()
            except Exception as e:
                logger.warning("ReAct LLM call failed: %s", e)
                if self.recorder:
                    self.recorder.record_tokens(count_tokens(prompt), 0)
                    self.recorder.finalize("", False, _REASON_ERROR)
                return "Agent encountered an error: {}".format(e)

            if self.recorder:
                self.recorder.record_tokens(count_tokens(prompt), count_tokens(response))

            if "Final Answer:" in response:
                idx = response.index("Final Answer:")
                answer = response[idx + len("Final Answer:"):].strip()
                logger.info("ReAct final answer (iterations=%d)", i + 1)
                if self.recorder:
                    self.recorder.add_step(_make_step(
                        i + 1, response, "", "", None, True, step_start))
                    self.recorder.finalize(answer, True, _REASON_FINAL)
                return answer

            if "Action:" in response:
                thought_idx = response.find("Thought:")
                action_idx = response.find("Action:")

                thought = response[thought_idx + 8:action_idx].strip() if thought_idx >= 0 else ""
                action_str = response[action_idx + 7:].strip()

                tool_call_rec = None
                try:
                    if "{" in action_str:
                        start = action_str.index("{")
                        end = action_str.rindex("}") + 1
                        action = json.loads(action_str[start:end])

                        tool_name = action.get("tool", "")
                        tool_args = action.get("args", {})

                        if tool_name in self.tools:
                            try:
                                observation = self.tools[tool_name](**tool_args)
                                if not isinstance(observation, str):
                                    observation = str(observation)
                                tool_call_rec = _make_tool_call(
                                    i + 1, tool_name, tool_args, observation,
                                    True, "", step_start)
                            except Exception as e:
                                observation = "Tool error: {}".format(e)
                                tool_call_rec = _make_tool_call(
                                    i + 1, tool_name, tool_args, observation,
                                    False, str(e), step_start)
                        else:
                            observation = "Unknown tool: {}".format(tool_name)
                            tool_call_rec = _make_tool_call(
                                i + 1, tool_name, tool_args, observation,
                                False, "unknown_tool", step_start)

                        scratchpad += (
                            "\nStep {}:\nThought: {}\nAction: {}\n"
                            "Observation: {}\n"
                        ).format(i + 1, thought, action_str, observation[:500])
                        logger.info("ReAct step %d: tool=%s", i + 1, tool_name)
                    else:
                        scratchpad += "\nStep {}: Could not parse action: {}\n".format(
                            i + 1, action_str)
                except json.JSONDecodeError as e:
                    scratchpad += "\nStep {}: Action parse error: {}\n".format(i + 1, e)
                if self.recorder:
                    self.recorder.add_step(_make_step(
                        i + 1, response, action_str,
                        scratchpad.split("Observation:")[-1][:200] if "Observation:" in scratchpad else "",
                        tool_call_rec, False, step_start))
            else:
                logger.info("ReAct no action/answer at step %d", i + 1)
                if self.recorder:
                    self.recorder.add_step(_make_step(
                        i + 1, response, "", "", None, False, step_start))
                    self.recorder.finalize(response, False, _REASON_NO_ACTION)
                return response

        logger.warning("ReAct max iterations (%d) reached", self.max_iterations)
        if self.recorder:
            self.recorder.finalize(
                "Agent reached max iterations.", False, _REASON_MAX_ITER)
        return "Agent reached max iterations. Last scratchpad:\n{}".format(scratchpad)


def _make_step(step, response, action, observation, tool_call, is_final, start):
    from eval.trace import TraceStep
    thought = ""
    if "Thought:" in response:
        t = response.index("Thought:") + 8
        a = response.find("Action:")
        thought = response[t:a].strip() if a > t else response[t:].strip()
    return TraceStep(
        step=step, thought=thought, action=action,
        observation=observation, tool_call=tool_call,
        is_final=is_final, latency_ms=(time.time() - start) * 1000,
    )


def _make_tool_call(step, name, args, result, success, error, start):
    from eval.trace import ToolCallRecord
    return ToolCallRecord(
        step=step, tool_name=name, args=args, result=str(result)[:500],
        success=success, error=error, latency_ms=(time.time() - start) * 1000,
    )
