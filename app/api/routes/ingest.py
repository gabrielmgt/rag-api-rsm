"""Ingest endpoint"""

from fastapi import APIRouter#, status
from app.core.logging import logger
from app.exceptions.exceptions import DuplicateDocumentException
from app.models.schemas import IngestRequest, IngestResponse
from app.core.ingest.ingest import document_from_content_or_url_and_trace
#from app.exceptions.http_exceptions import IngestionException

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
    except DuplicateDocumentException as e:
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
            message="An error occurred",
            chunks_created=0
        )
        #raise IngestionException(status.HTTP_500_INTERNAL_SERVER_ERROR,
        #                         "Query failed: Internal Server Error") from e
