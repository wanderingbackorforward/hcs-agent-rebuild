"""Task classification agent - main controller that orchestrates intent routing."""
import logging

from config.model_provider import create_chat_model
from config.constants import SharedState
from db.db_router import DatabaseRouter
from agents.task_classification import (
    StateManager,
    TaskClassifier,
    AgentRouter,
    UnrelatedHandler,
    ClassificationProcessor,
)

logger = logging.getLogger(__name__)


class TaskClassificationAgent:
    def __init__(self, environment_agent, knowledge_agent, db_router=None):
        self.environment_agent = environment_agent
        self.knowledge_agent = knowledge_agent
        self.llm = create_chat_model(temperature=0)
        self.db = db_router or DatabaseRouter()

        shared_state = SharedState()
        self.state_manager = StateManager(shared_state)
        self.task_classifier = TaskClassifier(self.llm)
        self.agent_router = AgentRouter(environment_agent, knowledge_agent, self.state_manager)
        self.unrelated_handler = UnrelatedHandler(self.state_manager)
        self.classification_processor = ClassificationProcessor(
            self.task_classifier,
            self.state_manager,
            self.agent_router,
            self.unrelated_handler,
            session_repo=self.db.session,
            llm=self.llm,
        )

        # Wire cross-agent callbacks
        if hasattr(self.environment_agent, "unrelated_callback"):
            self.environment_agent.unrelated_callback = self.handle_unrelated
        if hasattr(self.knowledge_agent, "set_unrelated_callback"):
            self.knowledge_agent.set_unrelated_callback(self.handle_unrelated_async)

        # Backward-compat state alias
        self.state = self.state_manager.state

    async def classify_task_stream(self, user_input: str, session_id: str = None):
        async for token in self.classification_processor.process_task_stream(
            user_input, session_id=session_id
        ):
            yield token

    async def classify_task(self, user_input: str, session_id: str = None) -> str:
        result = ""
        async for token in self.classify_task_stream(user_input, session_id=session_id):
            result += token
        return result

    async def handle_unrelated(self, user_input: str, session_id: str = None):
        logger.info(f"Re-classifying: {user_input}")
        result = ""
        async for token in self.classification_processor.process_task_stream(
            user_input, session_id=session_id
        ):
            result += token
        return result

    async def handle_unrelated_async(self, user_input: str, session_id: str = None):
        async for token in self.classification_processor.process_task_stream(
            user_input, session_id=session_id
        ):
            yield token

    def reset_conversation(self):
        self.classification_processor.reset_conversation()
