"""Short-term memory - manages context window with rolling summary.

Interview talking point: "Short-term memory uses a rolling summary — when
context window overflows, older turns are LLM-summarized into a compact
paragraph, keeping token budget under control while preserving key info."
"""
import logging
from typing import List
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)
DEFAULT_MAX_TURNS = 6
KEEP_RECENT = 4


class ShortTermMemory:
    """Manages short-term conversation context with rolling summary."""

    def __init__(self, llm=None, max_turns: int = DEFAULT_MAX_TURNS):
        self.llm = llm
        self.max_turns = max_turns
        self._messages: List[dict] = []
        self._summary: str = ""

    @property
    def messages(self) -> List[dict]:
        return list(self._messages)

    @property
    def summary(self) -> str:
        return self._summary

    def add_message(self, role: str, content: str):
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self.max_turns:
            self._compress()

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
        prompt = (
            "请将以下对话历史压缩为简洁摘要，保留关键信息"
            "（用户意图、已确认的事实、重要决策）。\n{}\n"
            "## 需要压缩的对话\n{}\n## 摘要（150字以内，中文）："
        ).format(existing, transcript)

        if self.llm:
            try:
                # Use sync invoke directly — works in both sync and async contexts.
                result = self.llm.invoke([HumanMessage(content=prompt)])
                self._summary = result.content.strip()
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
