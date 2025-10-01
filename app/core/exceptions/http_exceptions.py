"""Module with custom exceptions for our RAG app"""

from typing import Any
from fastapi import HTTPException, status

class IngestionException(HTTPException):
    """Base exception for document ingestion errors"""

    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: Any = None,
    ):
        super().__init__(status_code=status_code, detail=detail)


class QueryException(HTTPException):
    """Base exception for query errors"""

    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail: Any = None,
    ):
        super().__init__(status_code=status_code, detail=detail)