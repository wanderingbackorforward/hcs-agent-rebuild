"""Sparse retriever based on BM25."""
import re
import jieba
from typing import List, Tuple, Dict
from rank_bm25 import BM25Okapi
from rag.ingestion.storage.chroma_store import ChromaStore


class SparseRetriever:
    def __init__(self, store: ChromaStore = None):
        self.store = store or ChromaStore()
        self._documents = []
        self._metadatas = []
        self._bm25 = None
        self._build_index()

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9]", " ", text)
        return list(jieba.cut_for_search(text))

    def _build_index(self):
        results = self.store.collection.get(include=["documents", "metadatas"])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        if not docs:
            self._documents = []
            self._metadatas = []
            self._bm25 = None
            return
        self._documents = docs
        self._metadatas = metas
        tokenized = [self._tokenize(d) for d in docs]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5,
                 filters: Dict = None) -> List[Tuple[str, str, float, Dict]]:
        if not self._bm25:
            return []
        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        indexed = list(enumerate(scores))
        indexed.sort(key=lambda x: x[1], reverse=True)

        results = []
        all_ids = self.store.collection.get(include=["documents"]).get("ids", [])
        for idx, score in indexed[:top_k * 3]:
            meta = self._metadatas[idx] if idx < len(self._metadatas) else {}
            if filters:
                match = all(meta.get(k) == v for k, v in filters.items())
                if not match:
                    continue
            chunk_id = all_ids[idx] if idx < len(all_ids) else f"chunk_{idx}"
            results.append((chunk_id, self._documents[idx], float(score), meta))
            if len(results) >= top_k:
                break
        return results

    def refresh(self):
        self._build_index()
