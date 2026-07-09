"""Shared JSON parsing utilities for classification modules.

Both TaskClassifier and ClassificationProcessor need to extract and
parse JSON from LLM output.  Centralising the logic here avoids
divergence between the two implementations.
"""
import json
from typing import Optional


DEFAULT_CLASSIFICATION = {
    "intent_type": "knowledge_qa",
    "required_fields": {},
    "missing_fields": [],
    "keywords": [],
    "topic": "",
}


def extract_json_object(text: str) -> Optional[str]:
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


def parse_classification_json(text: str) -> dict:
    """Parse LLM output as classification JSON, falling back to defaults."""
    try:
        json_text = extract_json_object(text)
        if json_text:
            return json.loads(json_text)
    except Exception:
        pass
    return dict(DEFAULT_CLASSIFICATION)
