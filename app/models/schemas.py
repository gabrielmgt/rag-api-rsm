"""Module that provides Pydantic Models for use in /ingest and /query in api"""

from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, HttpUrl, model_validator
from typing_extensions import Self


class IngestResponse(BaseModel):
    status: str
    message: str
    chunks_created: int

class QueryRequest(BaseModel):
    question: str

class Source(BaseModel):
    page: Optional[int] = None
    text: str

class QueryResponse(BaseModel):
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
