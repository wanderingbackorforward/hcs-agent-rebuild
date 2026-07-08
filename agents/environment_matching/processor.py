"""Environment matching processor - multi-turn conversation flow orchestrator."""
import logging
from typing import AsyncGenerator, Dict, List

from agents.context_lock import clear_lock

logger = logging.getLogger(__name__)


class EnvironmentMatchingProcessor:
    REQUIRED_FIELDS = ["env_type", "components", "region"]

    def __init__(self, parser, message_builder, environment_service, probe_service, db_router):
        self.parser = parser
        self.message_builder = message_builder
        self.environment_service = environment_service
        self.probe_service = probe_service
        self.db = db_router

    def find_missing_fields(self, history: dict) -> list:
        missing = []
        for f in self.REQUIRED_FIELDS:
            value = history.get(f)
            if value is None or value == [] or value == "":
                missing.append(f)
        return missing

    def update_history_from_data(self, history: dict, data: dict) -> bool:
        for field in self.REQUIRED_FIELDS + ["service_status", "resource_usage"]:
            if data.get(field) is not None:
                history[field] = data[field]
        return len(self.find_missing_fields(history)) == 0

    async def process_step(
        self,
        user_input: str,
        session_id: str,
        history: dict,
    ) -> AsyncGenerator[str, None]:
        # Persist user message
        self.db.session.append_history(session_id, "user", user_input)

        ai_response = await self.parser.parse_stream(user_input)
        data = self.parser.parse_data(ai_response)

        if data.get("unrelated"):
            yield "当前正在收集环境条件。请提供环境类型、组件、区域等信息，或说“退出”结束。"
            return

        complete = self.update_history_from_data(history, data)

        if not complete:
            missing = self.find_missing_fields(history)
            reply = self.message_builder.build_question(missing)
            self.db.session.update_fields(session_id, history)
            self.db.session.append_history(session_id, "assistant", reply)
            yield reply
            return

        # Fields complete: match and validate
        self.db.session.update_fields(session_id, history)
        candidates = self.environment_service.match(history)
        yield self.message_builder.build_candidates(candidates)

        # Task fully done: one manual unlock (the single explicit clear site).
        clear_lock(self.db.session, session_id)

        if candidates:
            # Validate top candidate
            top = candidates[0]
            validation = self.probe_service.validate_environment(
                env_id=top["id"],
                required_components=history.get("components", []),
                session_id=session_id,
            )
            yield self.message_builder.build_validation(validation)
            self.db.session.append_history(session_id, "assistant", str(validation))
        else:
            self.db.session.append_history(session_id, "assistant", "未找到候选环境")

    def get_extracted_fields(self, session_id: str) -> dict:
        return self.db.session.get_fields(session_id)
