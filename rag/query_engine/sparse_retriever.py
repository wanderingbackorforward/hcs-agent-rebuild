"""Sparse retriever based on SQLite FTS5 full-text search.

Replaces the in-memory rank_bm25 approach. FTS5 provides persistent, incremental
full-text search — no full rebuild needed when documents are added or updated.

The interface (retrieve + refresh) is kept compatible with HybridSearch.
refresh() is now a no-op since FTS5 is always up-to-date.
"""
from typing import List, Tuple, Dict

from rag.ingestion.storage.fts_store import FTS5Store
from config.settings import app_settings


class SparseRetriever:
    def __init__(self, fts_store: FTS5Store = None):
        self.fts_store = fts_store or FTS5Store()

    def retrieve(self, query: str, top_k: int = app_settings.retrieval_top_k,
                 filters: Dict = None) -> List[Tuple[str, str, float, Dict]]:
        """Keyword search via FTS5. Returns (chunk_id, text, score, metadata)."""
        results = self.fts_store.search(query, top_k=top_k, filters=filters)
        # FTS5Store.search returns (chunk_id, doc_id, original_text, score, meta)
        # SparseRetriever.retrieve returns (chunk_id, text, score, meta)
        return [
            (chunk_id, text, score, meta)
            for chunk_id, doc_id, text, score, meta in results
        ]

    def refresh(self):
        """No-op for FTS5 — index is always up-to-date.

        Kept for backward compatibility with HybridSearch which calls
        self.sparse.refresh() before retrieval.
        """
        pass
