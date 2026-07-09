"""Main chat handler - routes user input through TaskClassificationAgent.

Creates a shared MemoryService per session so that all agents (environment,
knowledge_qa) share the same short-term, long-term, and task memory layers.

Session caches use an LRU + TTL eviction policy to prevent unbounded memory
growth in long-running processes.
"""
import logging
import time
import uuid
from collections import OrderedDict

from config.settings import app_settings
from config.model_provider import create_chat_model, create_embedding_model
from db.db_router import DatabaseRouter
from agents.memory.memory_service import MemoryService
from agents.memory.long_term_memory import MEMORY_COLLECTION
from agents.task_classification_agent import TaskClassificationAgent
from agents.environment_matching_agent import EnvironmentMatchingAgent
from agents.knowledge_qa_agent import KnowledgeQAAgent
from rag.ingestion.storage.chroma_store import ChromaStore

logger = logging.getLogger(__name__)

_session_agents: "OrderedDict[str, tuple]" = OrderedDict()
_session_memory: "OrderedDict[str, tuple]" = OrderedDict()


def _evict_expired(cache: OrderedDict, ttl: int):
    """Remove entries older than *ttl* seconds."""
    now = time.time()
    expired = [k for k, (_, ts) in cache.items() if now - ts > ttl]
    for k in expired:
        cache.pop(k, None)
        logger.debug("Evicted expired session: %s", k)


def _touch(cache: OrderedDict, key: str, max_size: int, ttl: int):
    """Move *key* to most-recently-used, evict expired + overflow entries."""
    _evict_expired(cache, ttl)
    if key in cache:
        cache.move_to_end(key)
    while len(cache) > max_size:
        evicted_key, _ = cache.popitem(last=False)
        logger.info("Evicted oldest session to stay under limit: %s", evicted_key)


def _get_memory_service(session_id: str) -> MemoryService:
    """Get or create a shared MemoryService for the session."""
    if session_id not in _session_memory:
        _touch(_session_memory, session_id,
               app_settings.session_cache_max_size, app_settings.session_cache_ttl)
        llm = create_chat_model(temperature=app_settings.llm_temperature)
        db_router = DatabaseRouter()
        try:
            embedder = create_embedding_model()
            store = ChromaStore(collection_name=MEMORY_COLLECTION)
        except Exception:
            embedder = None
            store = None
        _session_memory[session_id] = (MemoryService(
            session_id=session_id,
            llm=llm,
            session_repo=db_router.session,
            embedder=embedder,
            store=store,
        ), time.time())
    else:
        _session_memory[session_id] = (
            _session_memory[session_id][0], time.time()
        )
    return _session_memory[session_id][0]


def _get_agent(session_id: str):
    if session_id not in _session_agents:
        _touch(_session_agents, session_id,
               app_settings.session_cache_max_size, app_settings.session_cache_ttl)
        memory_service = _get_memory_service(session_id)
        env_agent = EnvironmentMatchingAgent(
            session_id=session_id, memory_service=memory_service,
        )
        knowledge_agent = KnowledgeQAAgent(
            session_id=session_id, memory_service=memory_service,
        )
        task_agent = TaskClassificationAgent(env_agent, knowledge_agent)
        _session_agents[session_id] = (task_agent, time.time())
    else:
        _session_agents[session_id] = (
            _session_agents[session_id][0], time.time()
        )
    return _session_agents[session_id][0]


async def process_user_input_stream(
    user_input: str, session_id: str = None, task_id: str = None,
):
    """Stream-process a user message and yield reply tokens."""
    sid = session_id or str(uuid.uuid4())
    agent = _get_agent(sid)
    async for token in agent.classify_task_stream(
        user_input, session_id=sid, task_id=task_id,
    ):
        yield token


async def process_user_input(
    user_input: str, session_id: str = None, task_id: str = None,
) -> str:
    """Non-streaming wrapper for chat."""
    result = ""
    async for token in process_user_input_stream(
        user_input, session_id, task_id=task_id,
    ):
        result += token
    return result
