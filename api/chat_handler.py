"""Main chat handler - routes user input through TaskClassificationAgent.

Creates a shared MemoryService per session so that all agents (environment,
knowledge_qa) share the same short-term, long-term, and task memory layers.
"""
import uuid

from config.model_provider import create_chat_model, create_embedding_model
from db.db_router import DatabaseRouter
from agents.memory.memory_service import MemoryService
from agents.memory.long_term_memory import MEMORY_COLLECTION
from agents.task_classification_agent import TaskClassificationAgent
from agents.environment_matching_agent import EnvironmentMatchingAgent
from agents.knowledge_qa_agent import KnowledgeQAAgent
from rag.ingestion.storage.chroma_store import ChromaStore

_session_agents = {}
_session_memory = {}


def _get_memory_service(session_id: str) -> MemoryService:
    """Get or create a shared MemoryService for the session."""
    if session_id not in _session_memory:
        llm = create_chat_model(temperature=0)
        db_router = DatabaseRouter()
        try:
            embedder = create_embedding_model()
            store = ChromaStore(collection_name=MEMORY_COLLECTION)
        except Exception:
            embedder = None
            store = None
        _session_memory[session_id] = MemoryService(
            session_id=session_id,
            llm=llm,
            session_repo=db_router.session,
            embedder=embedder,
            store=store,
        )
    return _session_memory[session_id]


def _get_agent(session_id: str):
    if session_id not in _session_agents:
        memory_service = _get_memory_service(session_id)
        env_agent = EnvironmentMatchingAgent(
            session_id=session_id, memory_service=memory_service,
        )
        knowledge_agent = KnowledgeQAAgent(
            session_id=session_id, memory_service=memory_service,
        )
        task_agent = TaskClassificationAgent(env_agent, knowledge_agent)
        _session_agents[session_id] = task_agent
    return _session_agents[session_id]


async def process_user_input_stream(user_input: str, session_id: str = None):
    """Stream-process a user message and yield reply tokens."""
    sid = session_id or str(uuid.uuid4())
    agent = _get_agent(sid)
    async for token in agent.classify_task_stream(user_input, session_id=sid):
        yield token


async def process_user_input(user_input: str, session_id: str = None) -> str:
    """Non-streaming wrapper for chat."""
    result = ""
    async for token in process_user_input_stream(user_input, session_id):
        result += token
    return result
