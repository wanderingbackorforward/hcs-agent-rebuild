"""FastAPI application entry point for HCS Agent Platform."""
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from services.knowledge_service import KnowledgeService
from services.environment_service import EnvironmentService
from db.db_router import DatabaseRouter
from api.core.exceptions import BusinessException, api_exception_handler, general_exception_handler
from web.routes import router as web_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def initialize_system():
    """Initialize database, knowledge base, and default environments."""
    try:
        logger.info("🚀 Initializing HCS Agent Platform...")
        db = DatabaseRouter()
        logger.info("📚 Initializing knowledge service...")
        knowledge_service = KnowledgeService(db)
        knowledge_service.initialize()
        logger.info("🖥️ Seeding default environments...")
        env_service = EnvironmentService(db)
        env_service.seed()
        logger.info("✅ System initialization complete.")
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        raise


def create_app() -> FastAPI:
    app = FastAPI(
        title="HCS 测试辅助 Agent 平台",
        description="HCS 测试环境匹配 + MCP 知识检索 Agent 平台",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_exception_handler(BusinessException, api_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    app.include_router(web_router)
    app.mount("/static", StaticFiles(directory="web/static"), name="static")

    @app.on_event("startup")
    async def startup_event():
        await initialize_system()

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
