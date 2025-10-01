"""RAG App main file"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.main import api_router
from app.core.metrics import setup_metrics
from app.core.logging import logger
from app.config.pydantic_settings import settings
from app.api.lifespan_setup import auto_ingest_base_documents

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events."""
    logger.info(
        "application_startup",
        project_name=settings.Project_name,
        version=settings.Version,
    )
    await auto_ingest_base_documents()
    yield
    logger.info("application_shutdown")

app = FastAPI(
    title=settings.Project_name,
    version=settings.Version,
    lifespan=lifespan
)

app.include_router(api_router)

setup_metrics(app)
