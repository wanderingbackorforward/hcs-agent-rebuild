"""MCP Prompt: rag_answer — standardized RAG answering prompt template.

Prompts are *predefined templates*: the host fills in arguments and
sends the rendered message to the LLM. This standardizes how the LLM
is instructed for RAG answering, ensuring consistent output format
across different callers.

Wraps the versioned prompt file prompts/rag_answer_v1.txt.
"""
import logging

from mcp import types

from prompts.loader import load_prompt

logger = logging.getLogger(__name__)

PROMPT_FILE = "rag_answer_v1.txt"

PROMPT_NAME = "rag-answer"
PROMPT_DESCRIPTION = (
    "RAG answer prompt: given retrieved context and a user query, "
    "instructs the LLM to answer concisely in Chinese."
)
PROMPT_ARGUMENTS = [
    types.PromptArgument(
        name="context",
        description="Retrieved knowledge context to ground the answer.",
        required=True,
    ),
    types.PromptArgument(
        name="query",
        description="The user's question.",
        required=True,
    ),
]


def rag_answer_handler(context: str, query: str) -> str:
    """Render the RAG answer prompt with the given context and query."""
    template = load_prompt(PROMPT_FILE)
    return template.format(context=context, query=query)


def register_prompt(protocol_handler) -> None:
    protocol_handler.register_prompt(
        name=PROMPT_NAME,
        description=PROMPT_DESCRIPTION,
        arguments=PROMPT_ARGUMENTS,
        handler=rag_answer_handler,
    )
