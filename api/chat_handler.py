"""Main chat handler - routes user input through TaskClassificationAgent."""
import uuid

from agents.task_classification_agent import TaskClassificationAgent
from agents.environment_matching_agent import EnvironmentMatchingAgent
from agents.knowledge_qa_agent import KnowledgeQAAgent

_session_agents = {}


def _get_agent(session_id: str):
    if session_id not in _session_agents:
        env_agent = EnvironmentMatchingAgent(session_id=session_id)
        knowledge_agent = KnowledgeQAAgent(session_id=session_id)
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
