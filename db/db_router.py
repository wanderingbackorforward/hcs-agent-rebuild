"""Database router: unified entry to all repositories."""
from db.base import SessionManager, run_lightweight_migrations
from db.models import Base
from db.repositories import (
    EnvironmentRepository,
    ValidationRepository,
    KnowledgeRepository,
    SessionRepository,
)


class DatabaseRouter:
    def __init__(self, db_path: str = None):
        self.session_manager = SessionManager(db_path)
        Base.metadata.create_all(self.session_manager.engine)
        run_lightweight_migrations(self.session_manager.engine)
        self._environment_repo = EnvironmentRepository(self.session_manager)
        self._validation_repo = ValidationRepository(self.session_manager)
        self._knowledge_repo = KnowledgeRepository(self.session_manager)
        self._session_repo = SessionRepository(self.session_manager)

    @property
    def environment(self) -> EnvironmentRepository:
        return self._environment_repo

    @property
    def validation(self) -> ValidationRepository:
        return self._validation_repo

    @property
    def knowledge(self) -> KnowledgeRepository:
        return self._knowledge_repo

    @property
    def session(self) -> SessionRepository:
        return self._session_repo

    def close(self):
        self.session_manager.close()
