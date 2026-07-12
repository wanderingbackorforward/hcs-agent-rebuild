#!/usr/bin/env python
"""Full index rebuild from raw document archive.

Reads all archived documents from the raw doc archive, re-chunks, re-embeds,
and rebuilds both Chroma vector index and FTS5 full-text index from scratch.

Usage:
    python scripts/rebuild_index.py

When to use:
    - Embedding model changed (vectors need regeneration)
    - Chunk strategy adjusted (re-split all documents)
    - FTS5 index corrupted or missing
    - Migration from rank_bm25 to FTS5 (one-time)

Prerequisites:
    - Raw documents must be archived (data/raw_docs/{doc_id}/content.txt)
    - Embedding service must be available (LLM_API_KEY or OPENAI_API_KEY)
"""
import sys
import os

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.ingestion.storage.raw_doc_archive import RawDocArchive
from rag.ingestion.pipeline import IngestionPipeline
from db.db_router import DatabaseRouter


def rebuild():
    archive = RawDocArchive()
    doc_ids = archive.list_archived()

    if not doc_ids:
        print("No archived documents found. Nothing to rebuild.")
        return 0

    print(f"Found {len(doc_ids)} archived documents.")
    print("Initializing pipeline...")

    db_router = DatabaseRouter()
    pipeline = IngestionPipeline(knowledge_repo=db_router.knowledge)

    print(f"Rebuilding index for {len(doc_ids)} documents...")
    rebuilt = 0
    failed = 0

    for doc_id in doc_ids:
        content = archive.read(doc_id)
        meta = archive.read_meta(doc_id)
        if not content:
            print(f"  SKIP (no content): {doc_id}")
            failed += 1
            continue
        try:
            pipeline.ingest_text(
                content=content,
                doc_id=doc_id,
                category=meta.get("category", "spec") if meta else "spec",
                title=meta.get("title") if meta else None,
                source=meta.get("source") if meta else None,
                force=True,
            )
            print(f"  OK: {doc_id}")
            rebuilt += 1
        except Exception as e:
            print(f"  FAIL: {doc_id} — {e}")
            failed += 1

    db_router.close()
    print(f"\nRebuild complete: {rebuilt} succeeded, {failed} failed.")
    return failed


if __name__ == "__main__":
    sys.exit(1 if rebuild() > 0 else 0)
