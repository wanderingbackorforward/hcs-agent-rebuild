"""Agent router - dispatches classified requests to the right Agent."""
from typing import AsyncGenerator

from config.constants import StateEnum


class AgentRouter:
    def __init__(self, environment_agent, knowledge_agent, state_manager):
        self.environment_agent = environment_agent
        self.knowledge_agent = knowledge_agent
        self.state_manager = state_manager

    async def route(self, intent_type: str, user_input: str, session_id: str = None) -> AsyncGenerator[str, None]:
        if intent_type == "environment_match":
            self.state_manager.set_state(StateEnum.ENVIRONMENT)
            async for token in self.environment_agent.process_stream(user_input, session_id=session_id):
                yield token
        elif intent_type == "knowledge_qa":
            self.state_manager.set_state(StateEnum.KNOWLEDGE)
            async for token in self.knowledge_agent.process_stream(user_input, session_id=session_id):
                yield token
        else:
            self.state_manager.set_state(StateEnum.OTHER)
            yield "我不太理解您的需求。您可以问我关于 HCS 测试环境匹配或技术规范查询的问题。"
