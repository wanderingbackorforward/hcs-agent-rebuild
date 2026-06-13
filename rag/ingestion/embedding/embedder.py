"""Embedding wrapper."""
from typing import List
from config.model_provider import create_embedding_model


class Embedder:
    def __init__(self):
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = create_embedding_model()
        return self._model

    def embed(self, text: str) -> List[float]:
        return self.model.embed_query(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)
