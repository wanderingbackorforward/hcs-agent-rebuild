"""Knowledge retriever - retrieves relevant documents using Hybrid Search."""
from services.knowledge_service import KnowledgeService


class KnowledgeRetriever:
    def __init__(self, knowledge_service: KnowledgeService = None):
        self.knowledge_service = knowledge_service or KnowledgeService()

    def retrieve(self, query: str, top_k: int = 5, filters: dict = None):
        self.knowledge_service.initialize()
        return self.knowledge_service.search(query, top_k=top_k, filters=filters)
