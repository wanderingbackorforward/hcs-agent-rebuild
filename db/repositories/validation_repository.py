"""Repository for validation records."""
from typing import List
from db.base import SessionManager
from db.models import ValidationRecord


class ValidationRepository:
    def __init__(self, session_manager: SessionManager):
        self._session_manager = session_manager

    def _session(self):
        return self._session_manager.get_session()

    def add(self, environment_id: int, session_id: str = None,
            probe_result: dict = None, matched_components: list = None) -> ValidationRecord:
        with self._session() as session:
            record = ValidationRecord(
                environment_id=environment_id,
                session_id=session_id,
                probe_result=probe_result or {},
                matched_components=matched_components or [],
            )
            session.add(record)
            session.commit()
            session.refresh(record)
            return record

    def list_by_session(self, session_id: str) -> List[ValidationRecord]:
        with self._session() as session:
            return session.query(ValidationRecord).filter(
                ValidationRecord.session_id == session_id
            ).order_by(ValidationRecord.created_at.desc()).all()
