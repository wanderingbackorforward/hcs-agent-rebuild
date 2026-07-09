"""Knowledge QA agent - main controller for RAG-based technical Q&A.

Chat history is persisted to SQLite via SessionRepository (table user_sessions).
In-process cache is a no-op now — persistence is the source of truth, so
process restarts do not lose conversation context.

U1+U2 upgrade: Integrated layered memory (short-term + long-term) and
context manager for token budget control.
"""
import logging
import uuid
from typing import List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from config.model_provider import create_chat_model, create_embedding_model
from config.settings import app_settings
from db.db_router import DatabaseRouter
from db.repositories.session_repository import SessionRepository
from services.knowledge_service import KnowledgeService
from agents.knowledge_qa import KnowledgeRetriever, ResponseGenerator
from agents.memory import ShortTermMemory, LongTermMemory, TaskMemory
from agents.memory.long_term_memory import MEMORY_COLLECTION
from agents.context_manager import ContextManager, count_tokens
from config.sse_protocol import SSEEvent
from cache.registry import get_semantic_cache
from rag.ingestion.storage.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


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

        # U2: Build context with token budget management.
        yield SSEEvent.status("retrieving", "正在检索知识库...")
        results = self.retriever.retrieve(user_query, top_k=app_settings.retrieval_top_k)

        # U3: Store retrieval results as intermediate results.
        self.task_memory.update_progress("query", user_query)
        self.task_memory.add_result("retrieval", {
            "doc_count": len(results),
            "top_docs": [
                {"doc_id": did, "title": meta.get("title", did), "score": score}
                for did, _, score, meta in results[:3]
            ],
        })

        # U1: Store user query in short-term memory.
        self.short_term_memory.add_message("user", user_query)

        # U2: Assemble context (includes long-term memory retrieval + STM summary).
        context = self.context_manager.build_prompt(
            self.SYSTEM_PROMPT, user_query
        )

        # Build RAG context from retrieved docs.
        rag_context = ""
        for i, (doc_id, text, score, meta) in enumerate(results[:app_settings.retrieval_top_k], 1):
            title = meta.get("title", doc_id)
            rag_context += f"[{i}] 来源：{title}\n{text}\n\n"

        # Generate answer with full context.
        prompt = f"""{context}

## 知识库检索结果
{rag_context}

## 答案（简洁、准确，使用中文）："""

        answer = ""
        yield SSEEvent.status("generating", "正在生成回答...")
        try:
            from langchain_core.messages import HumanMessage
            async for chunk in self.llm.astream([HumanMessage(content=prompt)]):
                answer += chunk.content
            answer = answer.strip()
            # Cache the successful answer for similar future queries.
            get_semantic_cache().set(user_query, answer)
        except Exception as e:
            logger.warning(f"LLM answer generation failed: {e}")
            answer = "根据知识库检索结果：\n" + "\n".join(
                f"- {meta.get('title', doc_id)}: {text[:200]}..."
                for doc_id, text, score, meta in results[:3]
            )

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
