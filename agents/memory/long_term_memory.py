"""Long-term memory - vector DB storage with RAG retrieval and memory gating.

Interview talking point: "Long-term memory uses a separate vector DB
collection with memory gating — LLM judges whether information is worth
persisting. Retrieval uses recency decay + semantic relevance weighted scoring."
"""
import json
import logging
import time
from typing import List, Dict, Optional, Tuple
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)
MEMORY_COLLECTION = "agent_long_term_memory"
DEFAULT_TTL = 30 * 24 * 3600
RECENCY_HALFLIFE = 7 * 24 * 3600
IMPORTANCE_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.15  # combined score below this -> not injected


class MemoryEntry:
    def __init__(self, content: str, memory_type: str = "fact",
                 importance: float = 0.5, timestamp: float = None,
                 metadata: dict = None):
        self.content = content
        self.memory_type = memory_type
        self.importance = importance
        self.timestamp = timestamp or time.time()
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "memory_type": self.memory_type,
            "importance": self.importance,
            "timestamp": self.timestamp,
            **self.metadata,
        }


class LongTermMemory:
    """Long-term memory backed by vector DB with memory gating."""

    def __init__(self, llm=None, embedder=None, store=None, reranker=None):
        self.llm = llm
        self.embedder = embedder
        self.store = store
        self.reranker = reranker
        self._collection = None
        self._init_collection()
        if self.reranker is None:
            try:
                from config.reranker_factory import create_reranker
                self.reranker = create_reranker()
            except Exception as e:
                logger.warning("Reranker init failed, using no-op: %s", e)
                from config.reranker_factory import NoOpReranker
                self.reranker = NoOpReranker()

    def _init_collection(self):
        if self.store is None:
            try:
                from rag.ingestion.storage.chroma_store import ChromaStore
                self.store = ChromaStore(collection_name=MEMORY_COLLECTION)
            except Exception as e:
                logger.warning("Failed to init memory collection: %s", e)

    def _embed(self, text: str) -> List[float]:
        """Embed text using the configured embedder.

        Adapts LangChain Embeddings (embed_query / embed_documents)
        to a uniform interface. Falls back to .embed() for custom embedders.
        """
        if hasattr(self.embedder, "embed_query"):
            return self.embedder.embed_query(text)
        if hasattr(self.embedder, "embed_documents"):
            return self.embedder.embed_documents([text])[0]
        return self.embedder.embed(text)

    def _judge_and_extract(self, content: str) -> Tuple[float, str, dict]:
        """Judge importance AND extract structured entities/relations.

        Returns (importance, memory_type, extracted) where extracted is
        a dict with 'entities' and 'relations' keys for structured retrieval.
        """
        if not self.llm:
            return 0.6, "fact", {}

        prompt = (
            '分析以下信息，判断是否值得长期记住，并提取结构化信息。\n'
            '评分标准：\n'
            '- 0.9-1.0：用户偏好、关键决策、重要事实（必须记住）\n'
            '- 0.5-0.8：可能有用的上下文（可以记住）\n'
            '- 0.0-0.4：寒暄、重复信息、无关内容（不需要记住）\n\n'
            '分类：fact / preference / decision / noise\n'
            '实体抽取：提取关键实体（如产品名、技术、人名、环境名等）\n\n'
            '## 信息内容\n{}\n\n'
            '## 输出格式（JSON）\n'
            '{{"importance": 0.X, "type": "fact|preference|decision|noise", '
            '"entities": ["实体1", "实体2"]}}'
        ).format(content)

        try:
            # Use sync invoke directly — works in both sync and async contexts.
            result = self.llm.invoke([HumanMessage(content=prompt)])
            text = result.content.strip()
            if "{" in text:
                start = text.index("{")
                end = text.rindex("}") + 1
                data = json.loads(text[start:end])
                importance = float(data.get("importance", 0.5))
                mem_type = data.get("type", "fact")
                entities = data.get("entities", [])
                if mem_type == "noise":
                    importance = max(0.0, importance)
                extracted = {"entities": entities} if entities else {}
                return importance, mem_type, extracted
        except Exception as e:
            logger.warning("Importance judging failed: %s", e)

        return 0.5, "fact", {}

    def _judge_importance(self, content: str) -> Tuple[float, str]:
        """Backward-compat wrapper: returns (importance, type) only."""
        importance, mem_type, _ = self._judge_and_extract(content)
        return importance, mem_type

    def store_memory(self, content: str, metadata: dict = None) -> bool:
        importance, mem_type, extracted = self._judge_and_extract(content)

        if importance < IMPORTANCE_THRESHOLD or mem_type == "noise":
            logger.info("Memory gated out (importance=%.2f, type=%s): %s",
                        importance, mem_type, content[:80])
            return False

        # Merge extracted entities into metadata for structured filtering.
        full_metadata = dict(metadata or {})
        if extracted:
            full_metadata.update(extracted)

        entry = MemoryEntry(
            content=content, memory_type=mem_type,
            importance=importance, metadata=full_metadata,
        )

        try:
            if self.store and self.embedder:
                embedding = self._embed(content)
                doc_id = "mem_{}".format(int(entry.timestamp * 1000))
                self.store.upsert(
                    doc_id=doc_id,
                    chunks=[content],
                    embeddings=[embedding],
                    metadatas=[entry.to_dict()],
                )
                logger.info("Memory stored (type=%s, importance=%.2f): %s",
                            mem_type, importance, content[:80])
                return True
            else:
                logger.warning("No store/embedder available")
                return False
        except Exception as e:
            logger.error("Failed to store memory: %s", e)
            return False

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if not self.store or not self.embedder:
            return []

        try:
            query_embedding = self._embed(query)
            # Stage 1: bi-encoder recall (fast, broad)
            results = self.store.query(query_embedding, top_k=top_k * 2)

            # Stage 2: cross-encoder rerank (precise, narrow)
            if self.reranker and results:
                results = self.reranker.rerank(query, results, top_k=top_k)

            # Stage 3: weighted scoring + confidence filter
            now = time.time()
            scored = []
            for doc_id, text, score, meta in results:
                # ChromaDB returns cosine distance; reranker returns similarity.
                # If score > 1.0 it's a reranker score (higher=better);
                # if 0..2 it's a distance (convert).
                if score <= 2.0:
                    relevance = max(0.0, 1.0 - score)
                else:
                    relevance = min(1.0, score / 10.0)
                mem_time = meta.get("timestamp", now)
                age = now - mem_time
                recency = 0.5 ** (age / RECENCY_HALFLIFE)
                importance = meta.get("importance", 0.5)
                combined = relevance * 0.5 + recency * 0.3 + importance * 0.2
                if combined < CONFIDENCE_THRESHOLD:
                    logger.debug("Memory filtered (score=%.3f < %.3f): %s",
                                 combined, CONFIDENCE_THRESHOLD, text[:60])
                    continue
                scored.append({
                    "content": text, "score": combined,
                    "relevance": relevance, "recency": recency,
                    "importance": importance,
                    "memory_type": meta.get("memory_type", "fact"),
                    "timestamp": mem_time,
                })
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored[:top_k]
        except Exception as e:
            logger.warning("Memory retrieval failed: %s", e)
            return []

    def get_context(self, query: str, top_k: int = 3) -> str:
        memories = self.retrieve(query, top_k=top_k)
        if not memories:
            return ""
        lines = ["[长期记忆]"]
        for m in memories:
            lines.append("- {} (类型: {}, 相关度: {:.2f})".format(
                m["content"], m["memory_type"], m["score"]))
        return "\n".join(lines)

    def clear(self):
        try:
            if self.store:
                self.store.delete_collection()
                self._init_collection()
        except Exception as e:
            logger.warning("Clear memory failed: %s", e)
