"""Module to setup prompt for RAG"""

#from langchain import hub
from langchain_core.prompts import ChatPromptTemplate

#prompt = hub.pull("rlm/rag-prompt")

prompt = ChatPromptTemplate.from_messages([
    ("system",  "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question. "
                "If you don't know the answer, just say that you don't know. "
                "If the context doesn't contain relevant information, say that you don't have enough information to answer the question" 
                "Use three sentences maximum and keep the answer concise.\n\n"
                "Context: {context} "),
    ("human",   "Question: {question}")
])
