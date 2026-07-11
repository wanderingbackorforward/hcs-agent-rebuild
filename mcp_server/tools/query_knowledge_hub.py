"""MCP Tool: query_knowledge_hub - hybrid search over HCS knowledge base."""
import json
import logging
from typing import Any, Dict, Optional

from mcp import types

from services.knowledge_service import KnowledgeService
from config.model_provider import create_chat_model
from config.settings import app_settings
from langchain_core.messages import HumanMessage
from mcp_server.errors import MCPError, format_error
from cache.registry import get_tool_cache
from prompts.loader import load_prompt

logger = logging.getLogger(__name__)


TOOL_NAME = "query_knowledge_hub"
TOOL_DESCRIPTION = "Search the HCS knowledge base for relevant documents using hybrid search (dense + BM25 + RRF)."
TOOL_INPUT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query or question.",
            "examples": ["HCS SDK 怎么接入？", "华为混合云存储类型"],
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
        "category": {
            "type": "string",
            "description": "Optional document category filter, such as sdk/spec/manual.",
        },
        "doc_id": {
            "type": "string",
            "description": "Optional document id filter to search inside a single document.",
            "examples": ["hcs-sdk-quickstart"],
        },
        "return_mode": {
            "type": "string",
            "description": "Return answer only, chunks only, or both.",
            "enum": ["answer", "chunks", "both"],
            "default": "both",
            "examples": ["both"],
        },
        "include_scores": {
            "type": "boolean",
            "description": "Whether to include retrieval scores in returned chunks.",
            "default": True,
        },
        "include_metadata": {
            "type": "boolean",
            "description": "Whether to include chunk metadata in returned chunks.",
            "default": True,
        },
        "max_chars_per_chunk": {
            "type": "integer",
            "description": "Maximum preview length for each returned chunk.",
            "default": 300,
            "minimum": 50,
            "maximum": 2000,
        },
    },
    "required": ["query"],
}
PROMPT_FILE = "rag_answer_v1.txt"


def _json_result(payload: Dict[str, Any]) -> types.CallToolResult:
    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=json.dumps(payload, ensure_ascii=False, indent=2),
            )
        ],
        isError=False,
    )


async def query_knowledge_hub_handler(
    query: str,
    top_k: int = app_settings.retrieval_top_k,
    collection: Optional[str] = None,
    category: Optional[str] = None,
    doc_id: Optional[str] = None,
    return_mode: str = "both",
    include_scores: bool = True,
    include_metadata: bool = True,
    max_chars_per_chunk: int = 300,
) -> types.CallToolResult:
    try:
        if return_mode not in {"answer", "chunks", "both"}:
            raise MCPError(
                error_type="invalid_input",
                message="return_mode must be one of: answer, chunks, both",
                details={"return_mode": return_mode},
            )
        if max_chars_per_chunk < 50 or max_chars_per_chunk > 2000:
            raise MCPError(
                error_type="invalid_input",
                message="max_chars_per_chunk must be between 50 and 2000",
                details={"max_chars_per_chunk": max_chars_per_chunk},
            )

        tool_cache = get_tool_cache()
        # Final-result cache: skip retrieval + LLM for repeated queries.
        # Key includes filters / mode so scoped searches don't collide.
        cached_text = tool_cache.get(
            query,
            "knowledge_hub",
            top_k=top_k,
            collection=collection,
            category=category,
            doc_id=doc_id,
            return_mode=return_mode,
            include_scores=include_scores,
            include_metadata=include_metadata,
            max_chars_per_chunk=max_chars_per_chunk,
        )
        if cached_text is not None:
            cached_payload = json.loads(cached_text)
            cached_payload["cache_hit"] = True
            return _json_result(cached_payload)

        service = KnowledgeService()
        service.initialize()
        filters = None
        effective_category = category or collection
        if effective_category or doc_id:
            filters = {}
            if effective_category:
                filters["category"] = effective_category
            if doc_id:
                filters["doc_id"] = doc_id
        results = service.search(query, top_k=top_k, filters=filters)

        payload: Dict[str, Any] = {
            "query": query,
            "answer": "",
            "retrieved_chunks": [],
            "filters_used": {
                "collection": collection,
                "category": effective_category,
                "doc_id": doc_id,
            },
            "top_k_used": top_k,
            "return_mode": return_mode,
            "cache_hit": False,
            "answer_generated_by": "none",
            "result_count": len(results),
        }

        if not results:
            payload["message"] = "未找到相关资料。"
            return _json_result(payload)

        context = ""
        for i, (doc_id, text, score, meta) in enumerate(results, 1):
            title = meta.get("title", doc_id)
            context += f"[{i}] 来源：{title}（score={score:.4f}）\n{text}\n\n"

        for i, (doc_id, text, score, meta) in enumerate(results, 1):
            chunk_doc_id = meta.get("doc_id", doc_id)
            item: Dict[str, Any] = {
                "rank": i,
                "doc_id": chunk_doc_id,
                "title": meta.get("title", chunk_doc_id),
                "text_preview": text[:max_chars_per_chunk],
            }
            if include_scores:
                item["score"] = round(float(score), 6)
            if include_metadata:
                item["metadata"] = {
                    "category": meta.get("category"),
                    "source": meta.get("source"),
                    "chunk_index": meta.get("chunk_index"),
                }
            payload["retrieved_chunks"].append(item)

        if return_mode in {"answer", "both"}:
            answer = ""
            try:
                llm = create_chat_model(temperature=app_settings.llm_temperature)
                prompt = load_prompt(PROMPT_FILE).format(context=context, query=query)
                full = ""
                async for chunk in llm.astream([HumanMessage(content=prompt)]):
                    full += chunk.content
                answer = full.strip()
                payload["answer_generated_by"] = "llm" if answer else "none"
            except Exception as e:
                logger.warning(f"LLM answer generation failed: {e}")
                answer = "根据知识库检索结果，请优先查看 returned chunks 中的候选证据。"
                payload["answer_generated_by"] = "fallback"
            payload["answer"] = answer
        else:
            payload["answer_generated_by"] = "skipped"

        if return_mode == "answer":
            payload["retrieved_chunks"] = []

        output_text = json.dumps(payload, ensure_ascii=False, indent=2)
        # Cache the final result for repeated queries.
        tool_cache.set(
            query,
            "knowledge_hub",
            output_text,
            top_k=top_k,
            collection=collection,
            category=category,
            doc_id=doc_id,
            return_mode=return_mode,
            include_scores=include_scores,
            include_metadata=include_metadata,
            max_chars_per_chunk=max_chars_per_chunk,
        )

        return _json_result(payload)
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
