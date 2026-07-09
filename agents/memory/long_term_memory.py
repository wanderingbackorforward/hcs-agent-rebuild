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

from config.settings import app_settings
from prompts.loader import load_prompt

logger = logging.getLogger(__name__)
MEMORY_COLLECTION = "agent_long_term_memory"
DEFAULT_TTL = app_settings.ltm_ttl
RECENCY_HALFLIFE = app_settings.ltm_recency_halflife
IMPORTANCE_THRESHOLD = app_settings.ltm_importance_threshold
CONFIDENCE_THRESHOLD = app_settings.ltm_confidence_threshold

_LTM_PROMPT_FILE = "ltm_judge_and_extract_v1.txt"


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
            return 0.75, "fact", {}  # optimistic default above threshold

        prompt = load_prompt(_LTM_PROMPT_FILE).format(content=content)

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

    def _mark_superseded(self, mem_type: str, query_embedding: List[float]):
        """Mark existing memories of the same type as superseded (conflict resolution).

        Newer information overrides older conflicting records. We find similar
        memories of the same type and set superseded=True so retrieve() skips them.
        """
        if not self.store or not self.embedder:
            return
        try:
            # Search for existing memories of the same type.
            results = self.store.query(
                query_embedding, top_k=10,
                filters={"memory_type": mem_type},
            )
            if not results:
                return
            ids_to_supersede = []
            for doc_id, text, score, meta in results:
                # Only supersede if not already superseded.
                if not meta.get("superseded", False):
                    ids_to_supersede.append(doc_id)
            if ids_to_supersede:
                self.store.update_metadata(ids_to_supersede, {"superseded": True})
                logger.info("Marked %d memories as superseded (type=%s)",
                            len(ids_to_supersede), mem_type)
        except Exception as e:
            logger.warning("Failed to mark superseded memories: %s", e)

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

                # Conflict resolution: mark same-type similar memories as superseded.
                self._mark_superseded(mem_type, embedding)

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

            # Stage 3: weighted scoring + confidence filter + conflict dedup
            now = time.time()
            scored = []
            for doc_id, text, score, meta in results:
                # Skip superseded memories (conflict resolution).
                if meta.get("superseded", False):
                    continue
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

            # Conflict dedup: group by memory_type, keep only the newest
            # highest-scoring record per type (avoids contradictory info).
            by_type = {}
            for item in scored:
                mtype = item["memory_type"]
                if mtype not in by_type:
                    by_type[mtype] = item
                else:
                    # Keep the one with higher combined score (recency already
                    # factored in, so newer + more relevant wins).
                    if item["score"] > by_type[mtype]["score"]:
                        by_type[mtype] = item
            scored = sorted(by_type.values(),
                            key=lambda x: x["score"], reverse=True)
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
