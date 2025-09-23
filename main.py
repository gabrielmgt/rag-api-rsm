import os
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl, Field, model_validator
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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from enum import Enum
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
import requests
import tempfile

from settings import Settings

settings = Settings()
app = FastAPI()

def initialize_langfuse():
    """
    Setup Langfuse instance here
    """
    return Langfuse(
        secret_key=settings.langfuse_secret_key,
        public_key=settings.langfuse_public_key,
        host=settings.langfuse_host,
    )

langfuse = initialize_langfuse()

def initialize_embeddings_model():
    """
    Setup embeddings models here, add more models, etc
    For our use case we don't really need more than HuggingFaceEmbeddings
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  
        model_kwargs={'device': 'cpu'},  
        encode_kwargs={'normalize_embeddings': True}
    )

embeddings = initialize_embeddings_model()

langfuse_callback_handler = CallbackHandler()

def initialize_vectorstore():
    """
    Setup Chroma vector store here 
    We consider an in-memory Chroma and Chroma running on a 
    separate container depending on the running mode defined 
    by environment variable ENV
    """
    if settings.ENV == "dev":
        return Chroma(
            embedding_function=embeddings, 
            persist_directory="./chroma_db",
            #host="localhost",
            #ssl=,
            #port=
            )
    elif settings.ENV == "prod":
        return Chroma(
            embedding_function=embeddings, 
            host=settings.chroma_host,
            port=settings.chroma_port,
            )


def initialize_chat_model():
    """
    Setup chat model here
    Google's model works best for our use case
    """
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",  
        google_api_key=settings.Google_API_Key,
        temperature=0.1
        )
    return model


chat_model = initialize_chat_model()
vectorstore = initialize_vectorstore()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)



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
    @classmethod
    def validate_input(cls, values: 'IngestRequest') -> 'IngestRequest':
        """Validate that either url or content is provided, but not both"""
        if values.url is None and values.content is None:
            raise ValueError('Either url or content must be provided')
        if values.url is not None and values.content is not None:
            raise ValueError('Provide either url or content, not both')
        
        return values
    
@app.get("/health")
def read_root():
    return {"status": "ok"}


@observe(name="document_splitting")
def split_documents_with_tracing(docs: str):
    """Do a tracing span for the method split_documents which has no callback for langfuse. no need for async because there's no external api calls"""
    chunks = text_splitter.split_documents(docs)
    return chunks

@observe(name="embedding_computation")
async def computate_embeddings_and_add_to_store(chunks: List[str]):
    """The method add_documents has embedding computations in it which we trace for using observe() because it doesn't accept the langfuse callback handler"""
    await vectorstore.aadd_documents(chunks, callbacks=[langfuse_callback_handler])

def load_document_from_content(content: str, document_type: DocumentType) -> List[Document]:
    """Load document from direct content"""
    metadata = {"source": "direct_input", "document_type": document_type.value}
    return [Document(page_content=content, metadata=metadata)]

@observe(name="load_document_from_url")
async def load_document_from_url(url: str, document_type: DocumentType) -> List[Document]:
    """Load document from URL based on document type"""
    try:
        if document_type == DocumentType.HTML:
            loader = WebBaseLoader([str(url)])
            docs = loader.load()
            return docs
            
        elif document_type == DocumentType.PDF:
            response = requests.get(str(url))
            response.raise_for_status()
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(response.content)
                tmp_file_path = tmp_file.name
            
            try:
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                return docs
            finally:
                os.unlink(tmp_file_path)
                
        elif document_type == DocumentType.TEXT:
            response = requests.get(str(url))
            response.raise_for_status()
            return [Document(page_content=response.text, metadata={"source": str(url)})]
            
        elif document_type == DocumentType.MARKDOWN:
            response = requests.get(str(url))
            response.raise_for_status()
            return [Document(page_content=response.text, metadata={"source": str(url), "type": "markdown"})]
            
    except Exception as e:
        raise ValueError(f"Failed to load document from URL: {e}")

@observe(name="document_ingestion")    
async def trace_ingest(request: IngestRequest):
    """Load document from URL if url is detected 
    Nest both chunk documents and embedding computations"""

    try:
        if request.url:
            docs = await load_document_from_url(request.url, request.document_type)
            for doc in docs:
                doc.metadata["source_url"] = str(request.url)
        elif request.content:
            #docs = [Document(page_content=request.content, metadata={"document_type": request.document_type})]
            docs = load_document_from_content(request.content, request.document_type)

        chunks = split_documents_with_tracing(docs)

        await computate_embeddings_and_add_to_store(chunks)
        return chunks
    except Exception as e:
        raise Exception(f"Document ingestion failed: {e}")

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

    model = chat_model

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
