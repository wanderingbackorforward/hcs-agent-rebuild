"""Knowledge QA agent - main controller for RAG-based technical Q&A.

Chat history is persisted to SQLite via SessionRepository (table user_sessions).
In-process cache is a no-op now — persistence is the source of truth, so
process restarts do not lose conversation context.
"""
import logging
import uuid
from typing import List, Optional

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage

from config.model_provider import create_chat_model
from db.db_router import DatabaseRouter
from db.repositories.session_repository import SessionRepository
from services.knowledge_service import KnowledgeService
from agents.knowledge_qa import KnowledgeRetriever, ResponseGenerator

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
    def __init__(self, session_id: str = None, db_router: Optional[DatabaseRouter] = None,
                 knowledge_service: Optional[KnowledgeService] = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.llm = create_chat_model(temperature=0)
        self.knowledge_service = knowledge_service or KnowledgeService(db_router=db_router)
        self.retriever = KnowledgeRetriever(self.knowledge_service)
        self.response_generator = ResponseGenerator(self.llm)
        # db_router.session is already a SessionRepository; use it directly.
        self._session_repo = (db_router or DatabaseRouter()).session
        self.chat_history = _SQLiteBackedHistory(self._session_repo, self.session_id)
        self._unrelated_callback = None
        self.initialized = False

    async def ensure_initialized(self):
        if not self.initialized:
            self.knowledge_service.initialize()
            self.initialized = True

    def set_unrelated_callback(self, callback):
        self._unrelated_callback = callback

    async def process_stream(self, user_query: str, session_id: str = None):
        await self.ensure_initialized()
        sid = session_id or self.session_id
        results = self.retriever.retrieve(user_query, top_k=5)
        answer = await self.response_generator.generate(user_query, results)
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
