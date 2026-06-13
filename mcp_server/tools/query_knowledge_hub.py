"""MCP Tool: query_knowledge_hub - hybrid search over HCS knowledge base."""
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from mcp import types

from services.knowledge_service import KnowledgeService
from config.model_provider import create_chat_model
from langchain_core.messages import HumanMessage
from mcp_server.errors import format_error

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).parent.parent.parent / "prompts"


def _load_prompt_template(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


TOOL_NAME = "query_knowledge_hub"
TOOL_DESCRIPTION = "Search the HCS knowledge base for relevant documents using hybrid search (dense + BM25 + RRF)."
TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query or question.",
        },
        "top_k": {
            "type": "integer",
            "description": "Maximum number of results to return.",
            "default": 5,
            "minimum": 1,
            "maximum": 20,
        },
        "collection": {
            "type": "string",
            "description": "Optional collection name to limit scope.",
        },
    },
    "required": ["query"],
}
PROMPT_FILE = "rag_answer_v1.txt"


async def query_knowledge_hub_handler(
    query: str,
    top_k: int = 5,
    collection: Optional[str] = None,
) -> types.CallToolResult:
    try:
        service = KnowledgeService()
        service.initialize()
        filters = None
        if collection:
            filters = {"category": collection}
        results = service.search(query, top_k=top_k, filters=filters)

        if not results:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="未找到相关资料。")],
                isError=False,
            )

        context = ""
        for i, (doc_id, text, score, meta) in enumerate(results, 1):
            title = meta.get("title", doc_id)
            context += f"[{i}] 来源：{title}（score={score:.4f}）\n{text}\n\n"

        answer = ""
        try:
            llm = create_chat_model(temperature=0)
            prompt = _load_prompt_template(PROMPT_FILE).format(context=context, query=query)
            full = ""
            async for chunk in llm.astream([HumanMessage(content=prompt)]):
                full += chunk.content
            answer = full.strip()
        except Exception as e:
            logger.warning(f"LLM answer generation failed: {e}")
            answer = ""

        lines = []
        if answer:
            lines.extend(["## AI Answer", "", answer, "", "---", ""])
        lines.append("## References")
        for i, (doc_id, text, score, meta) in enumerate(results, 1):
            title = meta.get("title", doc_id)
            lines.append(f"{i}. **{title}** (score={score:.4f})\n   {text[:200]}...")

        return types.CallToolResult(
            content=[types.TextContent(type="text", text="\n".join(lines))],
            isError=False,
        )
    except Exception as e:
        err = format_error(e, context="query_knowledge_hub")
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=err.to_text())],
            isError=True,
        )


def register_tool(protocol_handler) -> None:
    protocol_handler.register_tool(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=TOOL_INPUT_SCHEMA,
        handler=query_knowledge_hub_handler,
    )
