"""Knowledge service - initializes RAG pipeline and provides query interface."""
import logging

from db.db_router import DatabaseRouter
from config.settings import app_settings
from rag.ingestion.pipeline import IngestionPipeline
from rag.ingestion.storage.fts_store import FTS5Store
from rag.ingestion.storage.raw_doc_archive import RawDocArchive
from rag.query_engine.hybrid_search import HybridSearch
from rag.query_engine.sparse_retriever import SparseRetriever
from cache.registry import invalidate_semantic_cache, invalidate_tool_cache

logger = logging.getLogger(__name__)


class KnowledgeService:
    def __init__(self, db_router: DatabaseRouter = None):
        self.db = db_router or DatabaseRouter()
        # Share one FTS5Store between pipeline (writes) and retriever (reads)
        self.fts_store = FTS5Store(engine=self.db.session_manager.engine)
        self.archive = RawDocArchive()
        self.pipeline = IngestionPipeline(
            fts_store=self.fts_store,
            archive=self.archive,
            knowledge_repo=self.db.knowledge,
        )
        self.hybrid_search = HybridSearch(
            sparse=SparseRetriever(fts_store=self.fts_store)
        )
        self.initialized = False

    def initialize(self):
        """Seed default environments and knowledge documents."""
        if self.initialized:
            return
        logger.info("Initializing knowledge service...")
        self.db.environment.seed_defaults()
        try:
            self.pipeline.seed_defaults()
            # Migrate FTS5 from Chroma if FTS5 is empty but Chroma has data
            if self.fts_store.count() == 0:
                count = self.fts_store.migrate_from_chroma(self.pipeline.store)
                if count > 0:
                    logger.info("FTS5 migrated %d chunks from Chroma", count)
        except Exception as e:
            logger.warning(f"Knowledge seeding skipped (embedding unavailable): {e}")
        self.initialized = True
        logger.info("Knowledge service initialized.")

    def search(self, query: str, top_k: int = None, filters: dict = None):
        if top_k is None:
            top_k = app_settings.retrieval_top_k
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
        result = self.pipeline.ingest_text(content, doc_id, category, title, source)
        # Invalidate caches: new/updated documents must produce fresh retrieval
        # results and answers, not stale cached ones.
        invalidate_semantic_cache()
        invalidate_tool_cache()
        return result

    def delete_document(self, doc_id: str):
        """Delete a document from all stores and invalidate caches."""
        self.initialize()
        self.pipeline.delete_document(doc_id)
        invalidate_semantic_cache()
        invalidate_tool_cache()
