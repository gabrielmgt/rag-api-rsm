"""Module to setup prompt for RAG"""

from langchain import hub

prompt = hub.pull("rlm/rag-prompt")
