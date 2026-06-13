"""Knowledge service - initializes RAG pipeline and provides query interface."""
import logging

from db.db_router import DatabaseRouter
from rag.ingestion.pipeline import IngestionPipeline
from rag.query_engine.hybrid_search import HybridSearch

logger = logging.getLogger(__name__)


class KnowledgeService:
    def __init__(self, db_router: DatabaseRouter = None):
        self.db = db_router or DatabaseRouter()
        self.pipeline = IngestionPipeline()
        self.hybrid_search = HybridSearch()
        self.initialized = False

    def initialize(self):
        """Seed default environments and knowledge documents."""
        if self.initialized:
            return
        logger.info("Initializing knowledge service...")
        self.db.environment.seed_defaults()
        try:
            self.pipeline.seed_defaults()
            self._sync_to_sqlite()
        except Exception as e:
            logger.warning(f"Knowledge seeding skipped (embedding unavailable): {e}")
        self.initialized = True
        logger.info("Knowledge service initialized.")

    def _sync_to_sqlite(self):
        """Sync seeded knowledge documents into SQLite for record keeping."""
        for doc_id in self.pipeline.store.list_documents():
            content = self.pipeline.store.get_document_summary(doc_id) or ""
            self.db.knowledge.add(
                doc_id=doc_id,
                content=content[:1000],
                category="spec",
                title=doc_id,
                source="seed",
            )

    def search(self, query: str, top_k: int = 5, filters: dict = None):
        self.initialize()
        return self.hybrid_search.search(query, top_k=top_k, filters=filters)

    def list_documents(self):
        self.initialize()
        return self.pipeline.store.list_documents()

    def get_document_summary(self, doc_id: str):
        self.initialize()
        return self.pipeline.store.get_document_summary(doc_id)

    def ingest_text(self, content: str, doc_id: str = None, category: str = "spec",
                    title: str = None, source: str = None) -> str:
        self.initialize()
        return self.pipeline.ingest_text(content, doc_id, category, title, source)
