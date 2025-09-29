"""Health endpoint"""

from fastapi import APIRouter
from app.core.logging import logger

router = APIRouter()

@router.get("/health", tags=["health"])
def read_root():
    """
    Check health endpoint
    """
    logger.debug("health_check_request")
    return {"status": "ok"}
