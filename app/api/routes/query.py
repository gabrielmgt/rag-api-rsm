"""Query endpoint"""

from fastapi import APIRouter, status
from app.exceptions.http_exceptions import QueryException
from app.core.logging import logger
from app.models.schemas import QueryRequest, QueryResponse, Source
from app.core.langgraph.langgraph import graph
from app.core.observability.langfuse import langfuse_callback_handler

router = APIRouter()

@router.post("/query", response_model=QueryResponse, tags=["query"])
async def query_document(request: QueryRequest):
    """
    RAG Query endpoint
    """
    try:
        logger.info("query_received", question_length=len(request.question))


        logger.debug("generating_answer")

        response = await graph.ainvoke({"question": request.question}, # type: ignore
                                       config={"callbacks": [langfuse_callback_handler]})

        answer = response["answer"]
        sources = response["context"]

        formatted_sources = [
            Source(page=doc.metadata.get("page"), text=doc.page_content)
            #{"page": doc.metadata.get("page"), "text": doc.page_content}
            for doc in sources
        ]

        logger.debug("answer_generated", answer_length=len(answer))

        logger.info("query_success", 
                    answer_length=len(answer), 
                    sources_count=len(formatted_sources))
        return QueryResponse(answer=answer, sources=formatted_sources)
    except Exception as e:
        logger.error("query_failed", error=str(e))
        raise QueryException(status.HTTP_500_INTERNAL_SERVER_ERROR,
                                 "Query failed: Internal Server Error") from e
