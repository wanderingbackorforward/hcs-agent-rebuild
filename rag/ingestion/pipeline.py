"""Document ingestion pipeline."""
import os
import uuid
from typing import List
from rag.ingestion.chunking.chunker import TextChunker
from rag.ingestion.embedding.embedder import Embedder
from rag.ingestion.storage.chroma_store import ChromaStore


class IngestionPipeline:
    def __init__(self, store: ChromaStore = None, chunker: TextChunker = None,
                 embedder: Embedder = None):
        self.store = store or ChromaStore()
        self.chunker = chunker or TextChunker()
        self.embedder = embedder or Embedder()

    def ingest_text(self, content: str, doc_id: str = None, category: str = "spec",
                    title: str = None, source: str = None) -> str:
        doc_id = doc_id or uuid.uuid4().hex[:12]
        chunks = self.chunker.split(content)
        embeddings = self.embedder.embed_batch(chunks)
        metadatas = [
            {
                "category": category,
                "title": title or doc_id,
                "source": source or "inline",
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]
        self.store.upsert(doc_id, chunks, embeddings, metadatas)
        return doc_id

    def ingest_file(self, file_path: str, doc_id: str = None,
                    category: str = "spec", title: str = None) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        doc_id = doc_id or os.path.basename(file_path)
        return self.ingest_text(content, doc_id, category, title or doc_id, source=file_path)

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
        existing = set(self.store.list_documents())
        for d in defaults:
            if d["doc_id"] not in existing:
                self.ingest_text(
                    content=d["content"],
                    doc_id=d["doc_id"],
                    category=d["category"],
                    title=d["title"],
                    source="seed",
                )
