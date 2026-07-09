"""Dense retriever based on ChromaDB."""
from typing import List, Tuple, Dict
from rag.ingestion.storage.chroma_store import ChromaStore
from rag.ingestion.embedding.embedder import Embedder
from config.settings import app_settings


class DenseRetriever:
    def __init__(self, store: ChromaStore = None, embedder: Embedder = None):
        self.store = store or ChromaStore()
        self.embedder = embedder or Embedder()

    def retrieve(self, query: str, top_k: int = app_settings.retrieval_top_k,
                 filters: Dict = None) -> List[Tuple[str, str, float, Dict]]:
        query_embedding = self.embedder.embed(query)
        return self.store.query(query_embedding, top_k=top_k, filters=filters)
