"""Repository for user sessions (memory)."""
from typing import Optional
from db.base import SessionManager
from db.models import UserSession


class SessionRepository:
    def __init__(self, session_manager: SessionManager):
        self._session_manager = session_manager

    def _session(self):
        return self._session_manager.get_session()

    def get_or_create(self, session_id: str) -> UserSession:
        with self._session() as session:
            user_session = session.query(UserSession).filter(
                UserSession.session_id == session_id
            ).first()
            if not user_session:
                user_session = UserSession(session_id=session_id)
                session.add(user_session)
                session.commit()
                session.refresh(user_session)
            return user_session

    def update_fields(self, session_id: str, fields: dict):
        with self._session() as session:
            user_session = session.query(UserSession).filter(
                UserSession.session_id == session_id
            ).first()
            if user_session:
                current = user_session.extracted_fields or {}
                current.update(fields)
                user_session.extracted_fields = current
                session.commit()
                session.refresh(user_session)
            return user_session

    def append_history(self, session_id: str, role: str, content: str):
        with self._session() as session:
            user_session = session.query(UserSession).filter(
                UserSession.session_id == session_id
            ).first()
            if user_session:
                history = user_session.history or []
                history.append({"role": role, "content": content})
                user_session.history = history
                session.commit()

    def get_fields(self, session_id: str) -> dict:
        with self._session() as session:
            user_session = session.query(UserSession).filter(
                UserSession.session_id == session_id
            ).first()
            return user_session.extracted_fields if user_session else {}
