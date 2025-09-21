import os
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langfuse import Langfuse, observe
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Optional
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

app = FastAPI()

langfuse = Langfuse(
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
    host=os.environ.get("LANGFUSE_HOST"),
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  
    model_kwargs={'device': 'cpu'},  
    encode_kwargs={'normalize_embeddings': True}
)

langfuse_callback_handler = CallbackHandler()

vectorstore = Chroma(
    embedding_function=embeddings, 
    persist_directory="./chroma_db",
    #host="localhost",
    #ssl=,
    #port=
    )
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


class IngestRequest(BaseModel):
    content: str
    document_type: str

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
    
    
@app.get("/health")
def read_root():
    return {"status": "ok"}


@observe(name="document_splitting")
def split_documents_with_tracing(docs: str):
    """do a tracing span for the method split_documents which has no callback for langfuse. no need for async because there's no external api calls"""
    chunks = text_splitter.split_documents(docs)
    return chunks

@observe(name="embedding_computation")
async def computate_embeddings_and_add_to_store(chunks: List[str]):
    """add_documents has embedding computations in it which we trace for using observe() because it doesn't accept the langfuse callback handler"""
    await vectorstore.aadd_documents(chunks, callbacks=[langfuse_callback_handler])

@observe(name="document_ingestion")    
async def trace_ingest(request: IngestRequest):
    """nest both chunk documents and embedding computations"""
    docs = [Document(page_content=request.content, metadata={"document_type": request.document_type})]
    chunks = split_documents_with_tracing(docs)

    await computate_embeddings_and_add_to_store(chunks)
    return chunks

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    try:
        chunks = await trace_ingest(request)
        
        return IngestResponse(
            status="success",
            message=f"Successfully ingested document.",
            chunks_created=len(chunks)
        )
    except Exception as e:
        return IngestResponse(
            status="error",
            message=f"An error occurred: {e}",
            chunks_created=0
        )

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):


    template = """<|system|>
    You are a helpful AI assistant. Answer the question based only on the following context:
    {context}</s>
    <|user|>
    {question}</s>
    <|assistant|>
    """

    model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

    retriever = vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(template)

    retrieved_docs = await retriever.ainvoke(
        request.question,
        config={"callbacks": [langfuse_callback_handler]}
    )

    rag_chain = (
        {"context": RunnableLambda(lambda x: format_docs(retrieved_docs)), "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    answer = await rag_chain.ainvoke(
        request.question,
        config={"callbacks": [langfuse_callback_handler]}
        )

    sources = [
        Source(page=doc.metadata.get("page"), text=doc.page_content)
        for doc in retrieved_docs
    ]

    return QueryResponse(answer=answer, sources=sources)
