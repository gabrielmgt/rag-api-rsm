"""Custom exception handlers for the API"""

from app.exceptions.http_exceptions import IngestionException
from app.models.schemas import IngestResponse
from app.main import app

@app.exception_handler(IngestionException)
async def ingestion_exception_handler():
    """
    return IngestResponse with error status
    """
    return IngestResponse(
            status="error",
            message="An error occurred",
            chunks_created=0
        )
