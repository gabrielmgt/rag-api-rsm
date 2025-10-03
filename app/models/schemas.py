"""Module with Pydantic Models for ingest, query structures"""

from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, HttpUrl, model_validator
from typing_extensions import Self


class IngestResponse(BaseModel):
    """
    Pydantic Model for RAG API Responses
    """
    status: str
    message: str
    chunks_created: int

class QueryRequest(BaseModel):
    """
    Pydantic Model for Query Requests. Semantically, 
    the same model can be used for the API and the core application
    """
    question: str

class Source(BaseModel):
    """
    Pydantic Model to show associate text chunks to the source it was taken from.
    Used for query responses to show the client what context was used for the answer, 
    and from where.
    We validate that page isn't being used without a source to refer it to.
    """
    page: Optional[int] = None
    source: str
    text: str
    @model_validator(mode='after')
    def validate_input(self) -> Self:
        """Validate that either url or content is provided, but not both"""
        if self.page is not None and self.source is None:
            raise ValueError('Page field cannot be used if no source is provided')

        return self

class QueryResponse(BaseModel):
    """
    
    """
    answer: str
    sources: List[Source]

class DocumentType(str, Enum):
    PDF = "pdf"
    TEXT = "text"
    HTML = "html"
    MARKDOWN = "markdown"

class IngestRequest(BaseModel):
    content: Optional[str] = None
    url: Optional[HttpUrl] = None
    document_type: DocumentType

    @model_validator(mode='after')
    def validate_input(self) -> Self:
        """Validate that either url or content is provided, but not both"""
        if self.url is None and self.content is None:
            raise ValueError('Either url or content must be provided')
        if self.url is not None and self.content is not None:
            raise ValueError('Provide either url or content, not both')

        return self
