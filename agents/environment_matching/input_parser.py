"""Input parser - LLM-driven extraction of environment fields from user text."""
import json
import logging
import re
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class EnvironmentInputParser:
    """Extract environment requirements from natural language."""

    REQUIRED_FIELDS = ["env_type", "components", "region", "service_status"]

    def __init__(self, llm):
        self.llm = llm

    async def parse_stream(self, user_input: str, history: list = None) -> str:
        prompt = self._build_prompt(user_input, history)
        full = ""
        async for chunk in self.llm.astream([HumanMessage(content=prompt)]):
            full += chunk.content
        return full

    def parse_data(self, ai_response: str) -> dict:
        try:
            json_match = re.search(r"\{[^{}]*\}", ai_response)
            if json_match:
                return json.loads(json_match.group())
        except Exception:
            pass
        return {}

    def _build_prompt(self, user_input: str, history: list = None) -> str:
        history_text = ""
        if history:
            history_text = "\n".join(
                f"{'用户' if h.get('role') == 'user' else 'AI'}: {h.get('content', '')}"
                for h in history[-6:]
            )
            history_text = f"\n最近对话历史：\n{history_text}\n"

        return f"""你是 HCS 测试辅助平台的环境条件提取器。从用户输入中提取环境匹配所需的字段。
{history_text}
用户输入："{user_input}"

请返回 JSON（只返回 JSON）：
{{
    "env_type": "dev" | "test" | "staging" | null,
    "components": ["mysql", "redis", "kafka", ...] 或 [],
    "region": "beijing" | "shanghai" | null,
    "service_status": "available" | "busy" | "unknown" | null,
    "deploy_method": "docker" | "systemd" | "hcs" | null,
    "resource_usage": null,
    "unrelated": false
}}

规则：
- 只提取明确提到的信息，不要推测。
- components 为列表，如果用户提到多个组件请全部列出。
- deploy_method 枚举：docker(容器)/systemd(systemd 托管)/hcs(HCS 平台)。如未提及则为 null。
- 如果输入与环境匹配完全无关，设 unrelated=true 并其他字段为 null。"""
