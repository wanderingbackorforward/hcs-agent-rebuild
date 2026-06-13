"""Classification processor - orchestrates classify -> route -> respond pipeline."""
import json
import logging
import re
from typing import AsyncGenerator

logger = logging.getLogger(__name__)


class ClassificationProcessor:
    def __init__(self, classifier, state_manager, router, unrelated_handler):
        self.classifier = classifier
        self.state_manager = state_manager
        self.router = router
        self.unrelated_handler = unrelated_handler

    async def process_task_stream(self, user_input: str, session_id: str = None) -> AsyncGenerator[str, None]:
        raw = ""
        async for token in self.classifier.classify_stream(user_input):
            raw += token

        result = self._parse_json(raw)
        intent_type = result.get("intent_type", "knowledge_qa")
        logger.info(f"Classified as: {intent_type} (topic: {result.get('topic', 'N/A')})")

        if intent_type == "unrelated":
            reply = await self.unrelated_handler.handle(user_input)
            yield reply
        else:
            # Pass classification metadata to downstream agents via session context if available
            async for token in self.router.route(
                intent_type, user_input, session_id=session_id
            ):
                yield token

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

    def reset_conversation(self):
        self.state_manager.reset()
