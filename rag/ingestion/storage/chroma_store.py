"""ChromaDB vector store wrapper."""
import os
import uuid
from typing import List, Dict, Tuple, Optional
import chromadb
from chromadb.config import Settings


class ChromaStore:
    def __init__(self, collection_name: str = "hcs_knowledge",
                 persist_directory: str = "./data/chroma"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, doc_id: str, chunks: List[str], embeddings: List[List[float]],
               metadatas: List[Dict] = None):
        if not chunks:
            return
        ids = [f"{doc_id}_{uuid.uuid4().hex[:8]}" for _ in chunks]
        metadatas = metadatas or [{} for _ in chunks]
        for i, m in enumerate(metadatas):
            m["doc_id"] = doc_id
        self.collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding: List[float], top_k: int = 5,
              filters: Dict = None) -> List[Tuple[str, str, float, Dict]]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters,
            include=["documents", "distances", "metadatas"],
        )
        output = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        for i, doc_id in enumerate(ids):
            output.append((
                doc_id,
                docs[i],
                distances[i],
                metadatas[i] if metadatas else {},
            ))
        return output

    def list_documents(self) -> List[str]:
        results = self.collection.get(include=["metadatas"])
        doc_ids = set()
        for m in results.get("metadatas", []):
            doc_ids.add(m.get("doc_id", ""))
        return sorted(doc_ids)

    def get_document_summary(self, doc_id: str) -> Optional[str]:
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=["documents"],
        )
        docs = results.get("documents", [])
        return "\n".join(docs) if docs else None

    def update_metadata(self, ids: List[str], metadata_update: Dict):
        """Update metadata for existing documents by ID."""
        self.collection.update(ids=ids, metadatas=[metadata_update] * len(ids))

    def query_with_filter(self, query_embedding: List[float], top_k: int = 5,
                          where: Dict = None) -> List[Tuple[str, str, float, Dict]]:
        """Query with a metadata filter (alias for query with filters param)."""
        return self.query(query_embedding, top_k=top_k, filters=where)
