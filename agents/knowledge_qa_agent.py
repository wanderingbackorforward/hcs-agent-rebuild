"""Knowledge QA agent - main controller for RAG-based technical Q&A."""
import logging
import uuid

from langchain_core.chat_history import InMemoryChatMessageHistory

from config.model_provider import create_chat_model
from services.knowledge_service import KnowledgeService
from agents.knowledge_qa import KnowledgeRetriever, ResponseGenerator

logger = logging.getLogger(__name__)


class KnowledgeQAAgent:
    def __init__(self, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.llm = create_chat_model(temperature=0)
        self.knowledge_service = KnowledgeService()
        self.retriever = KnowledgeRetriever(self.knowledge_service)
        self.response_generator = ResponseGenerator(self.llm)

        self.chats_by_session_id = {}
        self.chat_history = self._get_chat_history(self.session_id)
        self._unrelated_callback = None
        self.initialized = False

    def _get_chat_history(self, sid: str) -> InMemoryChatMessageHistory:
        if sid not in self.chats_by_session_id:
            self.chats_by_session_id[sid] = InMemoryChatMessageHistory()
        return self.chats_by_session_id[sid]

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
