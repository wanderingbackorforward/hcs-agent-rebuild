"""Document ingestion pipeline."""
import hashlib
import logging
import os
from typing import List, Optional

from rag.ingestion.chunking.chunker import TextChunker
from rag.ingestion.chunking.quality_guard import ChunkQualityGuard
from rag.ingestion.embedding.embedder import Embedder
from rag.ingestion.storage.chroma_store import ChromaStore
from rag.ingestion.storage.raw_doc_archive import RawDocArchive
from rag.ingestion.storage.fts_store import FTS5Store
from config.chunker_factory import create_chunker, create_quality_guard
from config.vector_store_factory import create_vector_store

logger = logging.getLogger(__name__)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


class IngestionPipeline:
    def __init__(self, store: ChromaStore = None, chunker: TextChunker = None,
                 embedder: Embedder = None, guard: Optional[ChunkQualityGuard] = None,
                 archive: RawDocArchive = None, fts_store: FTS5Store = None,
                 knowledge_repo=None):
        self.store = store or create_vector_store()
        self.chunker = chunker or create_chunker()
        self.embedder = embedder or Embedder()
        self.guard = guard if guard is not None else create_quality_guard()
        self.archive = archive or RawDocArchive()
        self.fts_store = fts_store or FTS5Store()
        self.knowledge_repo = knowledge_repo  # Optional: SQLite metadata sync

    def ingest_text(self, content: str, doc_id: str = None, category: str = "spec",
                    title: str = None, source: str = None,
                    force: bool = False) -> str:
        effective_id = doc_id or _content_hash(content)

        # 1. Archive raw content (always — overwrites if exists)
        archive_path = self.archive.archive(
            effective_id, content, category, title, source
        )

        # 2. Idempotency: skip if already ingested (unless force)
        if not force and effective_id in self.store.list_documents():
            logger.info(f"ingest_text: skip duplicate doc_id={effective_id}")
            self._sync_metadata(effective_id, content, category, title, source, archive_path)
            return effective_id

        # 3. Delete old chunks if re-ingesting (force=True)
        if force:
            self.store.delete_by_doc_id(effective_id)
            self.fts_store.delete_by_doc_id(effective_id)

        # 4. Chunk
        chunks = self.chunker.split(content)
        # Rule-based pre-filter: only suspicious chunks pay an embedding cost
        chunks, assessments = self.guard.process(chunks, self.embedder)
        if self.guard.enabled and any(a.suspicious for a in assessments):
            flagged = sum(1 for a in assessments if a.suspicious)
            logger.info("ingest_text: guard flagged %d/%d chunks, %d segments after re-split",
                        flagged, len(assessments), len(chunks))

        # 5. Embed
        embeddings = self.embedder.embed_batch(chunks)
        metadatas = [
            {
                "category": category,
                "title": title or effective_id,
                "source": source or "inline",
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        # 6. Upsert to Chroma (returns generated chunk IDs)
        chunk_ids = self.store.upsert(effective_id, chunks, embeddings, metadatas)

        # 7. Insert into FTS5 index
        self.fts_store.upsert_chunks(chunk_ids, effective_id, chunks, metadatas)

        # 8. Sync metadata to SQLite
        self._sync_metadata(effective_id, content, category, title, source, archive_path)

        return effective_id

    def _sync_metadata(self, doc_id: str, content: str, category: str,
                       title: str, source: str, archive_path: str):
        """Sync document metadata to SQLite knowledge_documents table.

        This table serves MCP tools (list_collections, get_document_summary)
        and is NOT part of the RAG retrieval path.
        """
        if self.knowledge_repo is None:
            return
        try:
            self.knowledge_repo.add(
                doc_id=doc_id,
                content=content[:1000],
                category=category,
                title=title,
                source=source,
                archive_path=archive_path,
            )
        except Exception as e:
            logger.warning(f"SQLite metadata sync failed for {doc_id}: {e}")

    def ingest_file(self, file_path: str, doc_id: str = None,
                    category: str = "spec", title: str = None) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.ingest_text(
            content,
            doc_id or _content_hash(content),
            category,
            title or os.path.basename(file_path),
            source=file_path,
        )

    def delete_document(self, doc_id: str):
        """Delete a document from all stores: Chroma, FTS5, archive, SQLite."""
        self.store.delete_by_doc_id(doc_id)
        self.fts_store.delete_by_doc_id(doc_id)
        self.archive.delete(doc_id)
        if self.knowledge_repo is not None:
            try:
                self.knowledge_repo.delete(doc_id)
            except Exception as e:
                logger.warning(f"SQLite delete failed for {doc_id}: {e}")

    def seed_defaults(self):
        defaults = [
            {
                "doc_id": "hcs-sdk-quickstart",
                "category": "sdk",
                "title": "HCS SDK 快速入门",
                "content": (
                    "HCS SDK 是华为混合云平台的开发工具包。使用前需要在环境中配置 Access Key 和 Secret Key。"
                    "支持 Python 3.9 及以上版本。安装命令：pip install hcs-sdk。"
                    "初始化客户端时需要指定 region，例如 region='cn-north-4'。"
                ),
            },
            {
                "doc_id": "hcs-test-spec-env",
                "category": "spec",
                "title": "HCS 测试环境规范",
                "content": (
                    "执行 HCS 测试用例前，需要确认环境类型为 test 或 staging。"
                    "环境中必须包含 MySQL 5.7+ 或 8.0，Redis 5.0+，以及 Kafka 2.8+。"
                    "所有组件必须处于 available 状态，且端口可连通。"
                ),
            },
            {
                "doc_id": "hcs-manual-deploy",
                "category": "manual",
                "title": "HCS 部署手册",
                "content": (
                    "HCS 部署分为三个阶段：准备阶段、安装阶段、验收阶段。"
                    "准备阶段需要确认主机资源、网络规划和许可证。"
                    "安装阶段使用 hcs-deploy 工具一键部署。"
                    "验收阶段需要运行 smoke test 和回归测试。"
                ),
            },
        ]
        for d in defaults:
            # seed_defaults is itself idempotent: ingest_text short-circuits on duplicate.
            self.ingest_text(
                content=d["content"],
                doc_id=d["doc_id"],
                category=d["category"],
                title=d["title"],
                source="seed",
            )
