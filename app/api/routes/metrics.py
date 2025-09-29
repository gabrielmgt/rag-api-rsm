"""Metrics endpoint"""

import psutil
from fastapi import APIRouter, Response
from prometheus_client import generate_latest
from app.core.metrics import CPU_USAGE, MEMORY_USAGE

router = APIRouter()

@router.get("/metrics", tags=["metrics"])
def metrics():
    """Metrics endpoint declaration"""
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)
    return Response(generate_latest(), media_type="text/plain")
