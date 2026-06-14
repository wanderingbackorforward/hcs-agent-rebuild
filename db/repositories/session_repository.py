"""Repository for user sessions (memory).

Persists per-session state to the `user_sessions` table:
- extracted_fields: structured env-match fields (env_type / components / region / etc.)
- history: chat history as a list of {role, content} dicts

Both survive process restarts because the source of truth is the DB.
"""
from typing import List, Optional
from sqlalchemy.orm.attributes import flag_modified

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
                current = dict(user_session.extracted_fields or {})
                current.update(fields)
                user_session.extracted_fields = current
                flag_modified(user_session, "extracted_fields")
                session.commit()
                session.refresh(user_session)
            return user_session

    def append_history(self, session_id: str, role: str, content: str):
        with self._session() as session:
            user_session = session.query(UserSession).filter(
                UserSession.session_id == session_id
            ).first()
            if user_session:
                history = list(user_session.history or [])
                history.append({"role": role, "content": content})
                user_session.history = history
                # JSON column needs flag_modified for SQLAlchemy to detect
                # the change; otherwise the commit is a silent no-op.
                flag_modified(user_session, "history")
                session.commit()

    def get_fields(self, session_id: str) -> dict:
        with self._session() as session:
            user_session = session.query(UserSession).filter(
                UserSession.session_id == session_id
            ).first()
            return user_session.extracted_fields if user_session else {}

    def get_history(self, session_id: str) -> list:
        """Return a list of {role, content} dicts for the given session.
        Empty list if the session doesn't exist or has no history yet."""
        with self._session() as session:
            user_session = session.query(UserSession).filter(
                UserSession.session_id == session_id
            ).first()
            if user_session and user_session.history:
                return list(user_session.history)
            return []

    def clear_history(self, session_id: str) -> None:
        with self._session() as session:
            user_session = session.query(UserSession).filter(
                UserSession.session_id == session_id
            ).first()
            if user_session:
                user_session.history = []
                flag_modified(user_session, "history")
                session.commit()
