"""Ingest endpoint"""

from fastapi import APIRouter
from app.core.logging import logger
from app.core.exceptions.exceptions import DuplicateDocumentException
from app.models.schemas import IngestRequest, IngestResponse
from app.core.ingest.ingest import document_from_content_or_url_and_trace

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse, tags=["ingest"])
async def ingest_endpoint(request: IngestRequest):
    """
    Document Ingestion endpoint
    """
    logger.info("request_ingest_started", document_type=request.document_type)
    try:
        chunks = await document_from_content_or_url_and_trace(request)
        logger.info("request_ingest_completed", chunks_created=len(chunks))

        return IngestResponse(
            status="success",
            message="Successfully ingested document.",
            chunks_created=len(chunks)
        )
    except DuplicateDocumentException:
        logger.warning("ingest_request_duplicate", error=str(e))
        return IngestResponse(
            status="error",
            message="Document already exists",
            chunks_created=0
        )
    except Exception as e:
        logger.error("ingest_failed", error=str(e))
        return IngestResponse(
            status="error",
            message=f"An error occurred: {e}",
            chunks_created=0
        )
