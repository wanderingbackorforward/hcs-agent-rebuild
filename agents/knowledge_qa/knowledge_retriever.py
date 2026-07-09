"""Knowledge retriever - retrieves relevant documents using Hybrid Search."""
from services.knowledge_service import KnowledgeService
from config.settings import app_settings


class KnowledgeRetriever:
    def __init__(self, knowledge_service: KnowledgeService = None):
        self.knowledge_service = knowledge_service or KnowledgeService()

    def retrieve(self, query: str, top_k: int = app_settings.retrieval_top_k, filters: dict = None):
        self.knowledge_service.initialize()
        return self.knowledge_service.search(query, top_k=top_k, filters=filters)
