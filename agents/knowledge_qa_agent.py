"""Knowledge QA agent - main controller for RAG-based technical Q&A.

Chat history is persisted to SQLite via SessionRepository (table user_sessions).
In-process cache is a no-op now — persistence is the source of truth, so
process restarts do not lose conversation context.

U1+U2 upgrade: Integrated layered memory (short-term + long-term) and
context manager for token budget control.
"""
import logging
import re
import uuid
from typing import Dict, List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from config.model_provider import create_chat_model, create_embedding_model
from config.settings import app_settings
from db.db_router import DatabaseRouter
from db.repositories.session_repository import SessionRepository
from services.knowledge_service import KnowledgeService
from agents.knowledge_qa import KnowledgeRetriever, ResponseGenerator, KnowledgeToolBroker
from agents.memory import ShortTermMemory, LongTermMemory, TaskMemory
from agents.memory.long_term_memory import MEMORY_COLLECTION
from agents.context_manager import ContextManager, count_tokens
from config.sse_protocol import SSEEvent
from cache.registry import get_semantic_cache
from rag.ingestion.storage.chroma_store import ChromaStore

logger = logging.getLogger(__name__)
_DOC_ID_RE = re.compile(r"\b[hH][cC][sS][-_][a-zA-Z0-9][a-zA-Z0-9\-_]*\b")


class _SQLiteBackedHistory(BaseChatMessageHistory):
    """LangChain-compatible chat history backed by UserSession.history JSON.

    The on-disk JSON is the source of truth; the in-memory list mirrors it
    for the duration of one process. We rebuild from DB on every fetch,
    so restarts and multi-process all see the same conversation.
    """

    def __init__(self, repo: SessionRepository, session_id: str):
        self._repo = repo
        self._session_id = session_id
        # Ensure the row exists.
        self._repo.get_or_create(session_id)

    @property
    def messages(self) -> List:
        records = self._repo.get_history(self._session_id)
        return [
            HumanMessage(content=r["content"]) if r["role"] == "user" else AIMessage(content=r["content"])
            for r in records
        ]

    def add_message(self, message) -> None:
        role = "user" if isinstance(message, HumanMessage) else "ai"
        self._repo.append_history(self._session_id, role, message.content)

    def add_user_message(self, message: str) -> None:
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str) -> None:
        self.add_message(AIMessage(content=message))

    def clear(self) -> None:
        self._repo.clear_history(self._session_id)


class KnowledgeQAAgent:
    """Knowledge QA agent with layered memory and context management.

    Memory architecture (U1):
    - Short-term: recent N turns in context window, rolling summary for overflow.
    - Long-term: vector DB storage with memory gating, RAG retrieval.

    Context management (U2):
    - Token counting via tiktoken.
    - Overflow handling: compress -> truncate.
    - Layered assembly: system + memory + summary + recent + query.
    """

    SYSTEM_PROMPT = "你是 HCS 测试辅助助手。请严格根据知识库内容和对话历史回答用户问题。"

    def __init__(self, session_id: str = None, db_router: Optional[DatabaseRouter] = None,
                 knowledge_service: Optional[KnowledgeService] = None,
                 memory_service=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.llm = create_chat_model(temperature=app_settings.llm_temperature)
        self.knowledge_service = knowledge_service or KnowledgeService(db_router=db_router)
        self.retriever = KnowledgeRetriever(self.knowledge_service)
        self.response_generator = ResponseGenerator(self.llm)
        self.tool_broker = KnowledgeToolBroker()
        # db_router.session is already a SessionRepository; use it directly.
        self._session_repo = (db_router or DatabaseRouter()).session
        self.chat_history = _SQLiteBackedHistory(self._session_repo, self.session_id)
        self._unrelated_callback = None
        self.initialized = False

        # U4: Use shared MemoryService if provided, else create standalone.
        if memory_service:
            self.short_term_memory = memory_service.short_term
            self.long_term_memory = memory_service.long_term
            self.task_memory = memory_service.task_memory
        else:
            self.short_term_memory = ShortTermMemory(llm=self.llm)
            try:
                self.long_term_memory = LongTermMemory(
                    llm=self.llm,
                    embedder=create_embedding_model(),
                    store=ChromaStore(collection_name=MEMORY_COLLECTION),
                )
            except Exception as e:
                logger.warning(f"Long-term memory init failed, degrading to no-op: {e}")
                self.long_term_memory = LongTermMemory(llm=self.llm)
            self.task_memory = TaskMemory(
                session_repo=self._session_repo, session_id=self.session_id,
            )

        # U2: Context manager.
        self.context_manager = ContextManager(
            short_term_memory=self.short_term_memory,
            long_term_memory=self.long_term_memory,
            task_memory=self.task_memory,
        )

        # Load existing history into short-term memory.
        self._load_history_into_stm()

    def _load_history_into_stm(self):
        """Load existing chat history from SQLite into short-term memory."""
        try:
            records = self._session_repo.get_history(self.session_id)
            for record in records:
                self.short_term_memory.add_message(record["role"], record["content"])
        except Exception as e:
            logger.warning(f"Failed to load history into STM: {e}")

    async def ensure_initialized(self):
        if not self.initialized:
            self.knowledge_service.initialize()
            self.initialized = True

    def set_unrelated_callback(self, callback):
        self._unrelated_callback = callback

    def _extract_doc_id(self, query: str) -> Optional[str]:
        m = _DOC_ID_RE.search(query or "")
        return m.group(0) if m else None

    def _is_discovery_query(self, query: str) -> bool:
        query = (query or "").lower()
        hints = [
            "有哪些", "有什么", "哪些文档", "文档列表", "分类", "类别",
            "知识库", "目录", "集合", "清单", "列出",
        ]
        return any(h in query for h in hints)

    def _is_summary_query(self, query: str) -> bool:
        query = (query or "").lower()
        hints = ["摘要", "总结", "讲什么", "这篇", "该文档", "快速看看", "文档内容", "概述"]
        return bool(self._extract_doc_id(query)) or any(h in query for h in hints)

    def _is_list_only_query(self, query: str) -> bool:
        query = (query or "").lower()
        info_hints = ["怎么", "如何", "是什么", "原理", "要求", "说明", "介绍", "初始化"]
        return self._is_discovery_query(query) and not any(h in query for h in info_hints)

    def _keyword_hint(self, query: str) -> Optional[str]:
        query = (query or "").lower()
        for hint in ("sdk", "spec", "manual", "文档", "规范", "手册"):
            if hint in query:
                return hint
        return None

    def _format_list_answer(self, payload: Dict) -> str:
        collections = payload.get("collections", [])
        if not collections:
            return "当前知识库中没有可用集合。"
        collection = collections[0]
        lines = [f"当前知识库集合：{collection.get('name', 'hcs_knowledge')}"]
        if "doc_count" in collection:
            lines.append(f"文档总数：{collection['doc_count']}")
        categories = collection.get("categories") or []
        if categories:
            lines.append("分类：")
            lines.extend(
                f"- {item.get('name', 'unknown')}（{item.get('doc_count', 0)}）"
                for item in categories
            )
        sample_docs = collection.get("sample_docs") or []
        if sample_docs:
            lines.append("样本文档：")
            lines.extend(
                f"- {item.get('title') or item.get('doc_id')}"
                for item in sample_docs[:5]
            )
        return "\n".join(lines)

    def _format_query_answer(self, payload: Dict) -> str:
        answer = (payload.get("answer") or "").strip()
        if answer:
            return answer
        chunks = payload.get("retrieved_chunks") or []
        if not chunks:
            return payload.get("message") or "未找到相关资料。"
        lines = ["根据知识库检索结果："]
        for item in chunks[:3]:
            title = item.get("title") or item.get("doc_id") or "未知文档"
            preview = item.get("text_preview") or ""
            lines.append(f"- {title}: {preview}")
        return "\n".join(lines)

    def _format_summary_answer(self, summary_payload: Dict, query_payload: Optional[Dict] = None) -> str:
        lines = []
        if query_payload:
            answer = (query_payload.get("answer") or "").strip()
            if answer:
                lines.append(answer)
                lines.append("")
        title = summary_payload.get("title") or summary_payload.get("doc_id") or "目标文档"
        lines.append(f"补充文档摘要：{title}")
        lines.append(summary_payload.get("summary") or "暂无摘要。")
        return "\n".join(lines).strip()

    async def _answer_via_tools(self, user_query: str) -> tuple[str, Dict]:
        plan = []
        tool_results: Dict[str, Dict] = {}

        if self._is_discovery_query(user_query):
            plan.append("list_collections")
            tool_results["list_collections"] = await self.tool_broker.list_collections(
                include_stats=True,
                include_categories=True,
                include_doc_samples=True,
                keyword=self._keyword_hint(user_query),
            )
            if self._is_list_only_query(user_query):
                return self._format_list_answer(tool_results["list_collections"]), {
                    "path": "mcp_tools",
                    "plan": plan,
                    "tool_results": tool_results,
                }

        plan.append("query_knowledge_hub")
        tool_results["query_knowledge_hub"] = await self.tool_broker.query_knowledge_hub(
            query=user_query,
            top_k=app_settings.retrieval_top_k,
            return_mode="both",
            include_scores=True,
            include_metadata=True,
        )

        query_payload = tool_results["query_knowledge_hub"]
        doc_id = self._extract_doc_id(user_query)
        if not doc_id:
            chunks = query_payload.get("retrieved_chunks") or []
            if chunks:
                doc_id = chunks[0].get("doc_id")

        needs_summary = self._is_summary_query(user_query) or (
            not (query_payload.get("answer") or "").strip() and bool(doc_id)
        )
        if needs_summary and doc_id:
            plan.append("get_document_summary")
            tool_results["get_document_summary"] = await self.tool_broker.get_document_summary(
                doc_id=doc_id,
                max_chars=800,
                include_metadata=True,
                include_source=True,
                include_chunk_stats=True,
            )
            return self._format_summary_answer(
                tool_results["get_document_summary"], query_payload
            ), {
                "path": "mcp_tools",
                "plan": plan,
                "tool_results": tool_results,
            }

        return self._format_query_answer(query_payload), {
            "path": "mcp_tools",
            "plan": plan,
            "tool_results": tool_results,
        }

    async def _answer_via_legacy_path(self, user_query: str) -> tuple[str, Dict]:
        results = self.retriever.retrieve(user_query, top_k=app_settings.retrieval_top_k)
        self.task_memory.add_result("retrieval", {
            "doc_count": len(results),
            "top_docs": [
                {"doc_id": did, "title": meta.get("title", did), "score": score}
                for did, _, score, meta in results[:3]
            ],
        })

        context = self.context_manager.build_prompt(self.SYSTEM_PROMPT, user_query)
        rag_context = ""
        for i, (doc_id, text, score, meta) in enumerate(results[:app_settings.retrieval_top_k], 1):
            title = meta.get("title", doc_id)
            rag_context += f"[{i}] 来源：{title}\n{text}\n\n"

        prompt = f"""{context}

## 知识库检索结果
{rag_context}

## 答案（简洁、准确，使用中文）："""

        answer = ""
        try:
            async for chunk in self.llm.astream([HumanMessage(content=prompt)]):
                answer += chunk.content
            answer = answer.strip()
        except Exception as e:
            logger.warning(f"LLM answer generation failed: {e}")
            answer = "根据知识库检索结果：\n" + "\n".join(
                f"- {meta.get('title', doc_id)}: {text[:200]}..."
                for doc_id, text, score, meta in results[:3]
            )
        return answer, {
            "path": "legacy_fallback",
            "doc_count": len(results),
        }

    async def process_stream(self, user_query: str, session_id: str = None):
        await self.ensure_initialized()
        sid = session_id or self.session_id

        # U3: Set task type and track progress.
        self.task_memory.set_task("knowledge_qa")

        # Semantic cache: reuse answers for similar past queries.
        # On hit, skip retrieval + LLM generation + memory LLM calls; still
        # record the turn in STM and SQLite so conversation history is intact.
        cached = get_semantic_cache().get(user_query)
        if cached is not None:
            logger.info("Knowledge QA semantic cache hit, skipping retrieval+LLM")
            self.task_memory.update_progress("query", user_query)
            self.short_term_memory.add_message("user", user_query)
            self.short_term_memory.add_message("ai", cached)
            self.chat_history.add_user_message(user_query)
            self.chat_history.add_ai_message(cached)
            self.task_memory.add_result("answer", {"length": len(cached), "cached": True})
            self.task_memory.update_progress("answered", True)
            yield cached
            return

        self.task_memory.update_progress("query", user_query)
        self.short_term_memory.add_message("user", user_query)

        # Capability check: if MCP Server doesn't support tools, skip the
        # tool path entirely and go straight to legacy fallback.
        # This is the "pure client-side switch" — no glue code, just a boolean.
        tools_available = True
        try:
            flags = await self.tool_broker.ensure_initialized()
            tools_available = flags.tools_enabled
        except Exception as e:
            logger.warning("MCP Client initialize failed, treating as no-tools: %s", e)
            tools_available = False

        if not tools_available:
            logger.info("MCP Server tools not available, using legacy path directly")
            yield SSEEvent.status("fallback", "MCP Server 不支持 tools，回退旧链路...")
            answer, path_meta = await self._answer_via_legacy_path(user_query)
        else:
            try:
                yield SSEEvent.status("planning", "正在规划知识工具...")
                answer, path_meta = await self._answer_via_tools(user_query)
            except Exception as e:
                logger.warning("Tool-driven knowledge path failed, fallback to legacy: %s", e)
                yield SSEEvent.status("fallback", "知识工具失败，回退旧链路...")
                answer, path_meta = await self._answer_via_legacy_path(user_query)

        get_semantic_cache().set(user_query, answer)
        self.task_memory.add_result("tool_path", path_meta)

        # U1: Store AI response in short-term memory.
        self.short_term_memory.add_message("ai", answer)

        # U1: Refresh rolling summary every turn (not just on overflow).
        self.short_term_memory.refresh_summary()

        # U3: Store answer as intermediate result.
        self.task_memory.add_result("answer", {"length": len(answer)})
        self.task_memory.update_progress("answered", True)

        # U1: Memory gating - decide if this conversation is worth persisting.
        conversation_text = f"用户问: {user_query}\nAI答: {answer[:200]}"
        self.long_term_memory.store_memory(conversation_text)

        # Persist to SQLite.
        self.chat_history.add_user_message(user_query)
        self.chat_history.add_ai_message(answer)

        yield answer

    async def process(self, question: str, session_id: str = None) -> str:
        result = ""
        async for token in self.process_stream(question, session_id=session_id):
            result += token
        return result

    def reset(self):
        self.chat_history.clear()
        self.short_term_memory.clear()
        self.task_memory.archive()
