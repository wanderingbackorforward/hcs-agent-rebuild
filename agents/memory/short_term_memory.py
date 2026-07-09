"""Short-term memory - manages context window with rolling summary.

Interview talking point: "Short-term memory uses a rolling summary — when
context window overflows, older turns are LLM-summarized into a compact
paragraph, keeping token budget under control while preserving key info."
"""
import logging
from typing import List
from langchain_core.messages import HumanMessage

from config.settings import app_settings
from prompts.loader import load_prompt

logger = logging.getLogger(__name__)
DEFAULT_MAX_TURNS = app_settings.stm_max_turns
KEEP_RECENT = app_settings.stm_keep_recent

_STM_PROMPT_FILE = "stm_rolling_summary_v1.txt"


class ShortTermMemory:
    """Manages short-term conversation context with rolling summary."""

    def __init__(self, llm=None, max_turns: int = DEFAULT_MAX_TURNS):
        self.llm = llm
        self.max_turns = max_turns
        self._messages: List[dict] = []
        self._summary: str = ""
        self._sink_callback = None  # Called to sink key info to TaskMemory.

    @property
    def messages(self) -> List[dict]:
        return list(self._messages)

    @property
    def summary(self) -> str:
        return self._summary

    def set_sink_callback(self, callback):
        """Register a callback to sink key info to TaskMemory on compress/refresh.

        callback(summary: str) -> None
        """
        self._sink_callback = callback

    def add_message(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self.max_turns:
            self._compress()

    def refresh_summary(self):
        """Refresh rolling summary every turn (not just on overflow).

        Called after each AI response to keep the summary up-to-date.
        Generates a lightweight summary of the full conversation so far,
        so the ContextManager always has the latest state — even before
        overflow triggers _compress().
        """
        if not self.llm or len(self._messages) < 2:
            return  # Not enough context to summarize.

        # Build a transcript of all messages for a lightweight refresh.
        transcript = "\n".join(
            "User: {}".format(m["content"]) if m["role"] == "user"
            else "AI: {}".format(m["content"])
            for m in self._messages
        )

        existing = "\nPrevious summary:\n{}\n".format(self._summary) if self._summary else ""
        prompt = load_prompt(_STM_PROMPT_FILE).format(
            existing=existing, transcript=transcript,
        )

        try:
            result = self.llm.invoke([HumanMessage(content=prompt)])
            self._summary = result.content.strip()
            # Sink key info to TaskMemory if callback registered.
            if self._sink_callback and self._summary:
                self._sink_callback(self._summary)
        except Exception as e:
            logger.warning("Summary refresh failed: %s", e)

    def _compress(self):
        to_summarize = self._messages[:-KEEP_RECENT]
        self._messages = self._messages[-KEEP_RECENT:]
        if not to_summarize:
            return

        transcript = "\n".join(
            "User: {}".format(m["content"]) if m["role"] == "user"
            else "AI: {}".format(m["content"])
            for m in to_summarize
        )

        existing = "\nPrevious summary:\n{}\n".format(self._summary) if self._summary else ""
        prompt = load_prompt(_STM_PROMPT_FILE).format(
            existing=existing, transcript=transcript,
        )

        if self.llm:
            try:
                # Use sync invoke directly — works in both sync and async contexts.
                result = self.llm.invoke([HumanMessage(content=prompt)])
                self._summary = result.content.strip()
                # Sink key info to TaskMemory if callback registered.
                if self._sink_callback and self._summary:
                    self._sink_callback(self._summary)
            except Exception as e:
                logger.warning("Summary compression failed: %s", e)
                self._summary = (self._summary + " " + transcript)[-500:]
        else:
            self._summary = (self._summary + " " + transcript)[-500:]

    def get_context(self) -> str:
        parts = []
        if self._summary:
            parts.append("[对话摘要]\n{}".format(self._summary))
        if self._messages:
            parts.append("[近期对话]")
            for m in self._messages:
                speaker = "用户" if m["role"] == "user" else "AI"
                parts.append("{}: {}".format(speaker, m["content"]))
        return "\n".join(parts) if parts else ""

    def clear(self):
        self._messages.clear()
        self._summary = ""

    def to_dict(self) -> dict:
        return {"messages": self._messages, "summary": self._summary}

    def from_dict(self, data: dict):
        self._messages = data.get("messages", [])
        self._summary = data.get("summary", "")
