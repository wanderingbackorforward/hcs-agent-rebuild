"""Environment matching agent - main controller for multi-turn environment confirmation."""
import logging
import uuid

from langchain_core.chat_history import InMemoryChatMessageHistory

from config.model_provider import create_chat_model
from config.audit import audit_event, set_trace_context
from db.db_router import DatabaseRouter
from services.environment_service import EnvironmentService
from services.probe_service import ProbeService
from agents.environment_matching import (
    EnvironmentInputParser,
    EnvironmentMatchingProcessor,
    EnvironmentMessageBuilder,
)
from agents.memory.task_memory import TaskMemory

logger = logging.getLogger(__name__)


class EnvironmentMatchingAgent:
    def __init__(self, session_id: str = None, db_router: DatabaseRouter = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.db = db_router or DatabaseRouter()
        self.llm = create_chat_model(temperature=0)
        self.unrelated_callback = None

        self.input_parser = EnvironmentInputParser(self.llm)
        self.message_builder = EnvironmentMessageBuilder()
        self.environment_service = EnvironmentService(self.db)
        self.probe_service = ProbeService(self.db)
        self.processor = EnvironmentMatchingProcessor(
            self.input_parser,
            self.message_builder,
            self.environment_service,
            self.probe_service,
            self.db,
        )

        self.chats_by_session_id = {}
        self.chat_history = self._get_chat_history(self.session_id)
        self.history_by_session = {}

        # U3: Task memory - track env-matching progress and results.
        self.task_memory = TaskMemory(
            session_repo=self.db.session, session_id=self.session_id,
        )

    def _get_chat_history(self, sid: str) -> InMemoryChatMessageHistory:
        if sid not in self.chats_by_session_id:
            self.chats_by_session_id[sid] = InMemoryChatMessageHistory()
        return self.chats_by_session_id[sid]

    def _get_history(self, sid: str) -> dict:
        if sid not in self.history_by_session:
            stored = self.db.session.get_fields(sid)
            self.history_by_session[sid] = {
                "env_type": stored.get("env_type"),
                "components": stored.get("components", []),
                "region": stored.get("region"),
                "service_status": stored.get("service_status"),
                "deploy_method": stored.get("deploy_method"),
                "resource_usage": stored.get("resource_usage"),
            }
        return self.history_by_session[sid]

    async def process_stream(self, user_input: str, session_id: str = None):
        sid = session_id or self.session_id
        set_trace_context(session_id=sid, agent_name="environment_matching")
        audit_event(
            layer="worker",
            event_type="tool_decision",
            message="environment_matching agent: parse + match",
            data={"agent": "environment_matching", "session_id": sid, "query_length": len(user_input)},
        )
        # U3: Set task type and track query.
        self.task_memory.set_task("environment_matching")
        self.task_memory.update_progress("query", user_input)

        history = self._get_history(sid)
        # U3: Record collected fields into task memory.
        self.task_memory.update_progress("collected_fields", {
            k: v for k, v in history.items() if v
        })

        async for token in self.processor.process_step(user_input, sid, history):
            yield token

        # U3: Store matching result as intermediate result.
        self.task_memory.add_result("env_match_step", {
            "fields_complete": len(self.processor.find_missing_fields(history)) == 0,
        })

        audit_event(
            layer="worker",
            event_type="response_generated",
            message="environment_matching agent step done",
            data={"agent": "environment_matching"},
        )

    async def process(self, user_input: str, session_id: str = None) -> str:
        result = ""
        async for token in self.process_stream(user_input, session_id=session_id):
            result += token
        return result

    def reset(self):
        self.chat_history.clear()
        self.history_by_session = {}
        self.task_memory.archive()
