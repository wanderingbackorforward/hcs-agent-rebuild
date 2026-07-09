"""Hybrid search: dense + sparse + RRF fusion + optional rerank."""
from typing import List, Tuple, Dict
from rag.query_engine.dense_retriever import DenseRetriever
from rag.query_engine.sparse_retriever import SparseRetriever
from cache.registry import get_tool_cache
from config.reranker_factory import create_reranker, Reranker
from config.settings import app_settings


class HybridSearch:
    def __init__(self, dense: DenseRetriever = None, sparse: SparseRetriever = None,
                 dense_weight: float = 1.0, sparse_weight: float = 1.0,
                 reranker: Reranker = None):
        self.dense = dense or DenseRetriever()
        self.sparse = sparse or SparseRetriever(self.dense.store)
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.reranker = reranker if reranker is not None else create_reranker()

    def _rrf_fuse(self, dense_results: List[Tuple], sparse_results: List[Tuple],
                  k: int = 60) -> List[Tuple[str, str, float, Dict]]:
        scores = {}
        infos = {}

        for rank, (_, text, score, meta) in enumerate(dense_results):
            doc_id = meta.get("doc_id", "")
            if doc_id not in scores:
                scores[doc_id] = 0
                infos[doc_id] = (text, meta)
            scores[doc_id] += self.dense_weight / (k + rank + 1)

        for rank, (_, text, score, meta) in enumerate(sparse_results):
            doc_id = meta.get("doc_id", "")
            if doc_id not in scores:
                scores[doc_id] = 0
                infos[doc_id] = (text, meta)
            scores[doc_id] += self.sparse_weight / (k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        output = []
        for doc_id, score in ranked:
            text, meta = infos[doc_id]
            output.append((doc_id, text, score, meta))
        return output

    def search(self, query: str, top_k: int = app_settings.retrieval_top_k,
               filters: Dict = None) -> List[Tuple[str, str, float, Dict]]:
        # Tool cache: skip the dense+BM25+RRF+rerank pipeline for repeated
        # queries. Keyed by query+top_k+filters so scoped searches don't collide.
        tool_cache = get_tool_cache()
        cached = tool_cache.get(query, "hybrid_search", top_k=top_k, filters=filters)
        if cached is not None:
            return cached

        self.sparse.refresh()
        dense_results = self.dense.retrieve(query, top_k=top_k, filters=filters)
        sparse_results = self.sparse.retrieve(query, top_k=top_k, filters=filters)
        fused = self._rrf_fuse(dense_results, sparse_results)
        reranked = self.reranker.rerank(query, fused, top_k=top_k)

        tool_cache.set(query, "hybrid_search", reranked, top_k=top_k, filters=filters)
        return reranked
