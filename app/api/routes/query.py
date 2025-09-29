"""Query endpoint"""

from fastapi import APIRouter
from app.core.logging import logger
from app.models.schemas import QueryRequest, QueryResponse
from app.core.langgraph.langgraph import graph

router = APIRouter()

@router.post("/query", response_model=QueryResponse, tags=["query"])
async def query_document(request: QueryRequest):
    """
    RAG Query endpoint
    """
    logger.info("query_received", question_length=len(request.question))


    logger.debug("generating_answer")

    response = await graph.ainvoke({"question": request.question}) # type: ignore

    answer = response["answer"]
    sources = response["context"]

    logger.debug("answer_generated", answer_length=len(answer))

    logger.info("query_success", answer_length=len(answer), sources_count=len(sources))
    return QueryResponse(answer=answer, sources=sources)
