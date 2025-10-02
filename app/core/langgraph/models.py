"""Module for LangGraph State models"""

from langchain_core.documents import Document
from typing_extensions import List, TypedDict

class State(TypedDict):
    """LangGraph State class with enough fields for RAG"""
    question: str
    context: List[Document]
    answer: str
