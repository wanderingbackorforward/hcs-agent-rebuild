"""FastAPI application entry point for HCS Agent Platform."""
import logging

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from config.settings import app_settings
from services.knowledge_service import KnowledgeService
from services.environment_service import EnvironmentService
from db.db_router import DatabaseRouter
from api.core.exceptions import BusinessException, api_exception_handler, general_exception_handler
from api.middleware import TraceIdMiddleware
from web.routes import router as web_router

logging.basicConfig(
    level=getattr(logging, app_settings.log_level, logging.INFO)
)
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
        title=app_settings.app_name,
        description=app_settings.app_description,
        version=app_settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS: wildcard origins cannot be combined with credentials per spec.
    allow_credentials = app_settings.cors_allow_credentials
    if "*" in app_settings.cors_origins_list:
        allow_credentials = False
    app.add_middleware(
        CORSMiddleware,
        allow_origins=app_settings.cors_origins_list,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TraceIdMiddleware)

    app.add_exception_handler(BusinessException, api_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    app.include_router(web_router)
    app.mount("/static", StaticFiles(directory=app_settings.static_dir), name="static")

    @app.on_event("startup")
    async def startup_event():
        await initialize_system()

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=app_settings.host, port=app_settings.port)
