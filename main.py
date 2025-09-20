import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langfuse import Langfuse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Optional
from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import init_chat_model

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

vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
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
    
    
@app.post("/ingest", response_model=IngestResponse)
def ingest_document(request: IngestRequest):
    try:
        docs = [Document(page_content=request.content, metadata={"document_type": request.document_type})]
        chunks = text_splitter.split_documents(docs)
        
        vectorstore.add_documents(chunks)
        
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
def query_document(request: QueryRequest):
    langfuse_callback_handler = CallbackHandler()
    
    template = """<|system|>
    You are a helpful AI assistant. Answer the question based only on the following context:
    {context}</s>
    <|user|>
    {question}</s>
    <|assistant|>
    """
    prompt_template = ChatPromptTemplate.from_template(template)
    
    repo_id = "deepseek-ai/DeepSeek-R1-0528"

    
    model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

    retrieved_docs = vectorstore.similarity_search(
        request.question, 
        k=4,  
    )

    docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    prompt = prompt_template.invoke({
        "question": request.question, 
        "context": docs_content
    })

    model_response = model.invoke(
        prompt, 
        config={"callbacks": [langfuse_callback_handler]}
    )
    
    output_parser = StrOutputParser()

    
    answer = output_parser.invoke(model_response)
    
    sources = [
        Source(page=doc.metadata.get("page"), text=doc.page_content)
        for doc in retrieved_docs
    ]

    return QueryResponse(answer=answer, sources=sources)