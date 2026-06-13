"""Vector store factory: switch dense retrieval backends without touching call sites."""
import os
from typing import Optional

from rag.ingestion.storage.chroma_store import ChromaStore


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def create_vector_store(
    provider: Optional[str] = None,
    collection_name: Optional[str] = None,
) -> ChromaStore:
    """Build a vector store.

    Defaults read from env: VECTOR_STORE_PROVIDER (chroma, default chroma).
    Note: this MVP only ships ChromaDB; new backends (Qdrant / Milvus) plug in
    here as a future `if provider == "qdrant": return QdrantStore(...)` branch.
    """
    provider = (provider or _env_str("VECTOR_STORE_PROVIDER", "chroma")).lower()
    if provider != "chroma":
        raise ValueError(
            f"Unsupported VECTOR_STORE_PROVIDER={provider!r}. "
            "Only 'chroma' is wired in this build."
        )
    return ChromaStore(collection_name=collection_name) if collection_name else ChromaStore()
