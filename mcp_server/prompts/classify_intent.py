"""MCP Prompt: classify_intent — task classification prompt template.

Standardizes the intent classification instruction sent to the LLM.
The LLM analyzes user input and returns a JSON with intent_type,
required_fields, missing_fields, keywords, and topic.

Wraps the versioned prompt file prompts/classification_v1.txt.
"""
import logging
from pathlib import Path

from mcp import types

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"
PROMPT_FILE = "classification_v1.txt"

PROMPT_NAME = "classify-intent"
PROMPT_DESCRIPTION = (
    "Intent classification prompt: analyzes user input and returns "
    "a JSON with intent_type, required_fields, and keywords."
)
PROMPT_ARGUMENTS = [
    types.PromptArgument(
        name="user_input",
        description="The user's input text to classify.",
        required=True,
    ),
    types.PromptArgument(
        name="history_text",
        description="Optional conversation history for context.",
        required=False,
    ),
]


def _load_template() -> str:
    return (PROMPTS_DIR / PROMPT_FILE).read_text(encoding="utf-8")


def classify_intent_handler(user_input: str, history_text: str = "") -> str:
    """Render the classification prompt with the given arguments."""
    template = _load_template()
    return template.format(user_input=user_input, history_text=history_text)


def register_prompt(protocol_handler) -> None:
    protocol_handler.register_prompt(
        name=PROMPT_NAME,
        description=PROMPT_DESCRIPTION,
        arguments=PROMPT_ARGUMENTS,
        handler=classify_intent_handler,
    )
