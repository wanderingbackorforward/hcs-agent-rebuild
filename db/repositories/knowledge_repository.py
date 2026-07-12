"""Repository for knowledge documents."""
from typing import List, Optional
from db.base import SessionManager
from db.models import KnowledgeDocument


class KnowledgeRepository:
    def __init__(self, session_manager: SessionManager):
        self._session_manager = session_manager

    def _session(self):
        return self._session_manager.get_session()

    def add(self, doc_id: str, content: str, category: str,
            title: str = None, source: str = None,
            archive_path: str = None, version: int = 1,
            metadata_json: dict = None) -> KnowledgeDocument:
        with self._session() as session:
            doc = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.doc_id == doc_id
            ).first()
            if doc:
                doc.content = content
                doc.category = category
                doc.title = title
                doc.source = source
                doc.archive_path = archive_path
                doc.version = (doc.version or 0) + 1
                doc.metadata_json = metadata_json or {}
                doc.is_active = 1
            else:
                doc = KnowledgeDocument(
                    doc_id=doc_id,
                    title=title,
                    content=content,
                    category=category,
                    source=source,
                    archive_path=archive_path,
                    version=version,
                    metadata_json=metadata_json or {},
                )
                session.add(doc)
            session.commit()
            session.refresh(doc)
            return doc

    def get_by_doc_id(self, doc_id: str) -> Optional[KnowledgeDocument]:
        with self._session() as session:
            return session.query(KnowledgeDocument).filter(
                KnowledgeDocument.doc_id == doc_id,
                KnowledgeDocument.is_active == 1,
            ).first()

    def list_all(self, category: str = None) -> List[KnowledgeDocument]:
        with self._session() as session:
            query = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.is_active == 1
            )
            if category:
                query = query.filter(KnowledgeDocument.category == category)
            return query.all()

    def list_categories(self) -> List[str]:
        with self._session() as session:
            rows = session.query(KnowledgeDocument.category).filter(
                KnowledgeDocument.is_active == 1
            ).distinct().all()
            return [r[0] for r in rows]

    def delete(self, doc_id: str):
        """Hard-delete a knowledge document by doc_id."""
        with self._session() as session:
            session.query(KnowledgeDocument).filter(
                KnowledgeDocument.doc_id == doc_id
            ).delete()
            session.commit()

    def deactivate(self, doc_id: str):
        """Soft-delete: mark as inactive instead of removing the row."""
        with self._session() as session:
            doc = session.query(KnowledgeDocument).filter(
                KnowledgeDocument.doc_id == doc_id
            ).first()
            if doc:
                doc.is_active = 0
                session.commit()
