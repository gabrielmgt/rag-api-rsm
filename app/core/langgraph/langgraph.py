"""Module for LangGraph RAG Graph functions invoke"""


from langgraph.graph import START, StateGraph
from app.core.langgraph.models import State
from app.services.vectorstore import vector_store
from app.core.chat_model.prompt import prompt
from app.core.chat_model.llm import llm
from app.core.logging import logger


def retrieve(state: State):
    """LangGraph retrieve step node for RAG"""
    logger.debug("retrieving_relevant_documents")
    retrieved_docs = vector_store.similarity_search(state["question"])
    logger.debug("documents_retrieved", count=len(retrieved_docs))
    return {"context": retrieved_docs}



def generate(state: State):
    """LangGraph generation step node for RAG"""
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
