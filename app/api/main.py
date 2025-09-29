from fastapi import APIRouter

from app.api.routes import health, ingest, metrics, query


api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(ingest.router)
api_router.include_router(metrics.router)
api_router.include_router(query.router)
