"""SQLite FTS5 full-text index for sparse retrieval.

Replaces the in-memory rank_bm25 approach. FTS5 provides persistent, incremental
full-text search without the O(N) full rebuild that rank_bm25 required.

Design:
- Pre-tokenize with jieba (Chinese word segmentation), join tokens with spaces
- FTS5's unicode61 tokenizer splits on whitespace, indexing each jieba token as a term
- Store original_text as UNINDEXED column for single-query retrieval (no Chroma round-trip)
- chunk_id is the join key back to Chroma metadata
- Metadata fields (doc_id, category, title, source) are UNINDEXED for filtering

Schema:
    CREATE VIRTUAL TABLE chunk_fts USING fts5(
        chunk_id UNINDEXED,
        doc_id UNINDEXED,
        tokenized_text,          -- indexed, jieba pre-tokenized
        original_text UNINDEXED,  -- for retrieval, not searchable
        category UNINDEXED,
        title UNINDEXED,
        source UNINDEXED
    )
"""
import logging
import re
from typing import List, Tuple, Dict, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

# Lazy-load jieba (first call takes ~1s for dictionary loading)
_jieba_loaded = False


def _ensure_jieba():
    global _jieba_loaded
    if not _jieba_loaded:
        import jieba
        jieba.initialize()
        _jieba_loaded = True


def _tokenize(text: str) -> str:
    """Pre-tokenize text with jieba, return space-joined tokens for FTS5."""
    _ensure_jieba()
    import jieba
    cleaned = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
    tokens = jieba.cut_for_search(cleaned)
    # Filter empty tokens, join with spaces
    return " ".join(t.strip() for t in tokens if t.strip())


class FTS5Store:
    """SQLite FTS5-backed full-text index for chunk search."""

    def __init__(self, engine=None):
        if engine is None:
            from db.base import SessionManager
            engine = SessionManager().engine
        self._engine = engine
        self._init_table()

    def _init_table(self):
        with self._engine.connect() as conn:
            conn.execute(text("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts USING fts5(
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    tokenized_text,
                    original_text UNINDEXED,
                    category UNINDEXED,
                    title UNINDEXED,
                    source UNINDEXED
                )
            """))
            conn.commit()

    def insert_chunk(self, chunk_id: str, doc_id: str, text_content: str,
                     category: str = "", title: str = "", source: str = ""):
        """Insert a single chunk into the FTS5 index."""
        tokenized = _tokenize(text_content)
        with self._engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO chunk_fts
                    (chunk_id, doc_id, tokenized_text, original_text, category, title, source)
                VALUES
                    (:chunk_id, :doc_id, :tokenized_text, :original_text, :category, :title, :source)
            """), {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "tokenized_text": tokenized,
                "original_text": text_content,
                "category": category,
                "title": title,
                "source": source,
            })
            conn.commit()

    def upsert_chunks(self, chunk_ids: List[str], doc_id: str,
                      chunks: List[str], metadatas: List[Dict]):
        """Batch insert chunks into FTS5.

        Deletes existing entries for doc_id first (if any), then inserts all
        new chunks. This ensures re-ingestion replaces old index entries.
        """
        # Delete old entries for this doc_id
        self.delete_by_doc_id(doc_id)

        if not chunks:
            return

        with self._engine.connect() as conn:
            for i, (chunk_id, text_content) in enumerate(zip(chunk_ids, chunks)):
                meta = metadatas[i] if i < len(metadatas) else {}
                tokenized = _tokenize(text_content)
                conn.execute(text("""
                    INSERT INTO chunk_fts
                        (chunk_id, doc_id, tokenized_text, original_text, category, title, source)
                    VALUES
                        (:chunk_id, :doc_id, :tokenized_text, :original_text, :category, :title, :source)
                """), {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "tokenized_text": tokenized,
                    "original_text": text_content,
                    "category": meta.get("category", ""),
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                })
            conn.commit()

    def delete_by_doc_id(self, doc_id: str):
        """Delete all FTS5 entries for a doc_id."""
        with self._engine.connect() as conn:
            conn.execute(text(
                "DELETE FROM chunk_fts WHERE doc_id = :doc_id"
            ), {"doc_id": doc_id})
            conn.commit()

    def search(self, query: str, top_k: int = 5,
               filters: Dict = None) -> List[Tuple[str, str, str, float, Dict]]:
        """Search chunks by keyword. Returns (chunk_id, doc_id, original_text, score, metadata).

        score is positive (higher = more relevant), derived from FTS5's bm25 rank.
        """
        tokenized_query = _tokenize(query)
        if not tokenized_query.strip():
            return []

        # Build MATCH expression: wrap each token in quotes for safety
        match_expr = " ".join(f'"{t}"' for t in tokenized_query.split())

        sql = """
            SELECT chunk_id, doc_id, original_text, title, source, category, rank
            FROM chunk_fts
            WHERE chunk_fts MATCH :match
        """
        params: Dict = {"match": match_expr}

        if filters:
            for key, value in filters.items():
                if key in ("category", "doc_id", "source", "title"):
                    sql += f" AND {key} = :filter_{key}"
                    params[f"filter_{key}"] = value

        sql += " ORDER BY rank LIMIT :limit"
        params["limit"] = top_k

        try:
            with self._engine.connect() as conn:
                rows = conn.execute(text(sql), params).fetchall()
        except Exception as e:
            logger.warning("FTS5 search failed: %s", e)
            return []

        output = []
        for row in rows:
            chunk_id, doc_id, orig_text, title, source, category, rank = row
            meta = {
                "doc_id": doc_id,
                "title": title,
                "source": source,
                "category": category,
            }
            # FTS5 rank is negative (more negative = more relevant); negate for positive score
            score = -rank if rank is not None else 0.0
            output.append((chunk_id, doc_id, orig_text, score, meta))
        return output

    def count(self) -> int:
        """Return total number of indexed chunks."""
        with self._engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM chunk_fts")).scalar()
            return result or 0

    def clear(self):
        """Remove all entries from the FTS5 index."""
        with self._engine.connect() as conn:
            conn.execute(text("DELETE FROM chunk_fts"))
            conn.commit()

    def migrate_from_chroma(self, chroma_store):
        """Populate FTS5 index from existing Chroma chunks.

        Used for one-time migration when upgrading from rank_bm25 to FTS5.
        Reads all chunks from Chroma and inserts them into FTS5.
        """
        chunks = chroma_store.get_all_chunks()
        if not chunks:
            logger.info("FTS5 migration: no chunks in Chroma, skipping")
            return 0

        # Clear existing FTS5 entries first
        self.clear()

        count = 0
        with self._engine.connect() as conn:
            for chunk_id, text_content, meta in chunks:
                tokenized = _tokenize(text_content)
                conn.execute(text("""
                    INSERT INTO chunk_fts
                        (chunk_id, doc_id, tokenized_text, original_text, category, title, source)
                    VALUES
                        (:chunk_id, :doc_id, :tokenized_text, :original_text, :category, :title, :source)
                """), {
                    "chunk_id": chunk_id,
                    "doc_id": meta.get("doc_id", ""),
                    "tokenized_text": tokenized,
                    "original_text": text_content,
                    "category": meta.get("category", ""),
                    "title": meta.get("title", ""),
                    "source": meta.get("source", ""),
                })
                count += 1
            conn.commit()

        logger.info("FTS5 migration: inserted %d chunks from Chroma", count)
        return count
