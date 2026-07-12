"""ChromaDB vector store wrapper."""
import os
import uuid
from typing import List, Dict, Tuple, Optional
import chromadb
from chromadb.config import Settings

from config.settings import app_settings


class ChromaStore:
    def __init__(self, collection_name: str = None,
                 persist_directory: str = None):
        self.collection_name = collection_name or app_settings.knowledge_collection
        self.persist_directory = persist_directory or app_settings.chroma_persist_dir
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": app_settings.chroma_distance},
        )

    def upsert(self, doc_id: str, chunks: List[str], embeddings: List[List[float]],
               metadatas: List[Dict] = None) -> List[str]:
        """Upsert chunks and return the generated chunk IDs."""
        if not chunks:
            return []
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
        return ids

    def query(self, query_embedding: List[float], top_k: int = None,
              filters: Dict = None) -> List[Tuple[str, str, float, Dict]]:
        if top_k is None:
            top_k = app_settings.retrieval_top_k
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

    def delete_by_doc_id(self, doc_id: str):
        """Delete all chunks belonging to a doc_id."""
        self.collection.delete(where={"doc_id": doc_id})

    def get_chunks_by_doc_id(self, doc_id: str) -> List[Tuple[str, str, Dict]]:
        """Retrieve all (chunk_id, text, metadata) for a doc_id."""
        results = self.collection.get(
            where={"doc_id": doc_id},
            include=["documents", "metadatas"],
        )
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        return list(zip(ids, docs, metas))

    def get_all_chunks(self) -> List[Tuple[str, str, Dict]]:
        """Retrieve all chunks — used for FTS5 initial migration."""
        results = self.collection.get(include=["documents", "metadatas"])
        ids = results.get("ids", [])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        return list(zip(ids, docs, metas))

    def update_metadata(self, ids: List[str], metadata_update: Dict):
        """Update metadata for existing documents by ID."""
        self.collection.update(ids=ids, metadatas=[metadata_update] * len(ids))

    def query_with_filter(self, query_embedding: List[float], top_k: int = None,
                          where: Dict = None) -> List[Tuple[str, str, float, Dict]]:
        """Query with a metadata filter (alias for query with filters param)."""
        return self.query(query_embedding, top_k=top_k, filters=where)
