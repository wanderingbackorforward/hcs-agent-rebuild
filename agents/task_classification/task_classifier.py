"""Task classifier - uses LLM to classify user intent for HCS platform."""
import json
import logging
import re
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class TaskClassifier:
    """Classify user intent into environment_match, knowledge_qa, or unrelated."""

    def __init__(self, llm):
        self.llm = llm

    async def classify(self, user_input: str, history: list = None) -> dict:
        full = ""
        async for token in self.classify_stream(user_input, history):
            full += token
        return self._parse_json(full)

    async def classify_stream(self, user_input: str, history: list = None) -> AsyncGenerator[str, None]:
        prompt = self._build_prompt(user_input, history)
        async for chunk in self.llm.astream([HumanMessage(content=prompt)]):
            yield chunk.content

    def _build_prompt(self, user_input: str, history: list = None) -> str:
        history_text = ""
        if history:
            history_text = "\n".join(
                f"{'用户' if h.get('role') == 'user' else 'AI'}: {h.get('content', '')}"
                for h in history[-6:]
            )
            history_text = f"\n最近对话历史：\n{history_text}\n"

        return f"""你是 HCS 测试辅助 Agent 平台的意图分类器。分析用户输入，判断用户意图。
{history_text}
用户输入："{user_input}"

请返回 JSON（只返回 JSON，不要其他内容）：
{{
    "intent_type": "environment_match" | "knowledge_qa" | "unrelated",
    "required_fields": {{"env_type": "...", "components": ["..."], "region": "...", "service_status": "..."}},
    "missing_fields": [],
    "keywords": [],
    "topic": ""
}}

intent_type 取值：
- "environment_match"：用户想确认/查询/匹配 HCS 测试环境，包含环境类型、组件、服务状态、资源等。
- "knowledge_qa"：用户询问 SDK 文档、用户手册、内部测试规范、技术规范、接口说明等技术问题。
- "unrelated"：与 HCS 测试环境或技术规范完全无关的话题。

required_fields：如果 intent_type 是 environment_match，从用户输入中提取已明确的环境条件字段；否则留空。
missing_fields：environment_match 场景下，还未明确的关键字段（env_type, components, region, service_status, resource_usage）。
keywords：提取 2-5 个关键词。
topic：一句话概括核心诉求。"""

    def _parse_json(self, text: str) -> dict:
        try:
            json_text = self._extract_json_object(text)
            if json_text:
                return json.loads(json_text)
        except Exception:
            pass
        return {"intent_type": "knowledge_qa", "required_fields": {}, "missing_fields": [], "keywords": [], "topic": ""}

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        """Extract the outermost JSON object from text, supporting nested braces."""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None
