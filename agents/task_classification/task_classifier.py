"""Task classifier - uses LLM to classify user intent for HCS platform."""
import logging
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage

from agents.task_classification.json_utils import parse_classification_json
from prompts.loader import load_prompt

logger = logging.getLogger(__name__)


class TaskClassifier:
    """Classify user intent into environment_match, knowledge_qa, or unrelated."""

    def __init__(self, llm, prompt_file: str = "classification_v1.txt"):
        self.llm = llm
        self._prompt_template = load_prompt(prompt_file)

    async def classify(self, user_input: str, history: list = None) -> dict:
        full = ""
        async for token in self.classify_stream(user_input, history):
            full += token
        return parse_classification_json(full)

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
            history_text = f"\n最近对话历史:\n{history_text}\n"
        return self._prompt_template.format(history_text=history_text, user_input=user_input)
