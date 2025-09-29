"""RAG App main file"""

from fastapi import FastAPI
from app.api.main import api_router
from app.core.metrics import setup_metrics

app = FastAPI()

app.include_router(api_router)

setup_metrics(app)
