# RAG API
This is a FastAPI RAG microservice application with endpoints to provide document URLs for ingestion and answer queries using an LLM with support for full observabilty using LangFuse. 
## Considerations:
 - Using LangGraph graphs over LangChain chains: When attempting to use langchain chains for query processing, I found it rather complicated to keep a tidy LangFuse trace while also keeping an output for retrieved documents, as that means decoupling the retriever from the chain itself. LangGraph was a really good option here as the graph's invoke can take in the LangFuse callback handler and provide a neatly ordered trace with particularly relevant spans such as document retrieval and LLM inference. There are some other benefits that make this decision worthwhile, where personally I believe that the notation for graphs is easier to understand for someone who already has knowledge of classes in Python and graphs, and that LangGraph should definitely help add functionality to this application in the future: having support from the beginning for a graph style organization of operations is much better than having to manage multiple chains oneself. 

 - Using Grafana and Prometheus along LangChain: Some of the functionality that I divided along the structure of this project is expected to be able to work regardless of whether we want an HTTP API or another protocol. Namely, my hope is that all functionality for the RAG application itself be programmed in the core folder of this app, whereas HTTP endpoints and its logic be handled somewhere else, for the eventual case that we might want to use gRPC or something else. Therefore, I think it's not a bad idea to keep some metrics outside of LangFuse; rather, LangFuse should mainly care about tracing everything related to the RAG functionality itself, while something like Prometheus and Grafana can instead be used on metrics regarding HTTP requests.

 - My changes to the ingest API: I added an url field because having only a content field is ambiguous. An extra URL field gives two major benefits to our application. First, the backend service doesn't have to interpret the content field for an URL. This makes the endpoint easier to program and understand, and it also means we can write an application that allows for better modularity, as adding more fields just means adding more modules to our application, without having to rewrite logic that interprets the content field. Second, we don't have to process requests with massive texts in the content field. It makes more sense to download an URL instead. Third, clients don't have to deal with ambiguity over what goes in the content field. Nevertheless, I've left the content field to allow non-url inputs but have added a constraint to requests so that either only the url or the content field are used (not both) and would like to note that I would remove the content field as is in a real scenario. It might make sense in protocols other than HTTP, such as websockets, to have a similar content functionality.

 - My changes to the query API: I believe it is more reasonable to provide a document identifier instead of a page. This makes the source of the context to the answer easier to find, because a RAG application will likely have more than one document from which to pull information out of. A page field alone can hardly be used to find what document it belongs to. Moreover, not all documents may be paginated, and changing this field means that we can futureproof our API to new types of document. A trade-off for this however, means that we must track more metadata. Luckily, we already track metadata for sources in our idempotence check during ingestion, which is talked about below. Nevertheless, a page is not entirely useless, and may be left as an optional field. I have added a verification to the model to make sure that if page is given, that a source document must be present too.

 - Idempotence ingest check works for url and content fields in a different way and there is some discussion to be made of this: in the url case, I have decided to solve this by checking if an url has been entered already by using a query on the LangChain VectorStore, which is possible because chunks entered through an url ingest will have an url metadata associated to them. It is debatable if this is a good idea however; an url will not necessarily host the same content forever. Content checks work by calculating a checksum over the entire content string, which could also be an option for downloaded url contents, however it is unlikely that one would match content unless it is a file. Finally, it might be worthwhile to calculate checksums for every chunk, although just as checksums small changes in the string could greatly influence how the text splitter generates chunks. I do think that some interesting functionality could be achieved by having different versions of the same url through the use of tool calls to let the LLM filter retrieved documents by metadata. Regardless, there is a clear benefit in the current implementation which is not allowing the same url or content to be input several times in a short time, either through API usage error or user error.

 - Pydantic settings are used to set up two environments: prod and dev. I tried to focus on having as much of the api code be the same for both environments, where setup occurs strictly in app/config/pydantic_settings and involves credentials and dependencies: the vector store instance and eventually the instances of embeddings and LLM models.

 - Most modules have global variables initializated, where one alternative to look for would be to manage these as context for FastAPI, perhaps using lifespan before any requests are served.

 - Local environment uses an in-memory ChromaDB while Docker-compose prod environment has a container with ChromaDB. Thanks to LangChain, the way we interact with the ChromaDB VectorStore is identical in both environments.

 - Document loaders TODO: It would be nice to have a way to use documents loaders without constantly making new instances, but according to the LangChain API docs it seems these objects only receive file paths in the constructor

 - Current implementation only supports one of each: llm, embeddings model, graph, and vector_store. While this is enough for the required application, I would like to implement the ability to have multiple of each and somehow manage them in such a way that we can use the same module from both query and ingest (for the reason that we want to query the same vector store we ingested to)

 - Validation errors raised by Pydantic may not follow the API scheme. I believe it would be possible to change this using exception handlers on a future update.

 - Prompt choice: the prompt used for RAG encourages using the context to answer the question, answering that it doesn't know the answer if it's the case, or to say it doesn't have enough information if context was irrelevant. I went for a concise, 3 sentence maximum answer as shown in the LangChain RAG tutorial, but also because it makes more sense to keep the answer constrained to the content of the context, without necessity of elaborating further.

## Stack
- Python 3.13
- FastAPI
- LangChain / LangGraph / LangFuse (for RAG and observability)
- ChromaDB (for vector storage)
- LLM (Google GenAI)
- Embeddings (Google GenAI)
- Docker, Docker Compose

## API Reference

### Health Check
- **GET** `/health` - Returns 200 OK

### Ingest Endpoint
**POST** `/ingest`
```json
{
  "url": "https://example.com",  // OR
  "content": "<text>",
  "document_type": "<'html' | 'text' | 'pdf' | 'markdown'>"
}
```
returns:
```json
{
	"status": "'success'|'error'",
	"message": "<status message>",
	"chunks_created": `<number>`
}
```

### Query Endpoint
**POST** `/query`
```json
{
  "question": "<text>"
}
```
returns:
```json
{
  "answer": "<generated answer>",
  "sources": [
    {
      "text": "<passage text>",
      "source_document": "<url | 'user_input' | 'unknown'>",
      "page": "<number, optional>"
    }, 
		...
  ]
}
```
### Metrics Endpoint
- **GET** `/metrics` - Returns formatted metrics for Prometheus

## Dependencies
### Production Dependencies 
Refer to pyproject.toml dependencies and docker-compose.yml

### Development Dependencies
Refer to pyproject.toml dependencies and dev optional dependencies 

## Instructions
### Development Environment
Start with a virtual environment. I recommend conda:
1. Make a conda environment:
   ```bash
   conda create --prefix ./conda-env python=3.13
   ```
2. Activate the environment:
   ```bash
   conda activate ./conda-env
   ```
3. Install requirements:
   ```bash
   pip install -e .
   ```
4. Set environment variables in **.env.dev**. You may use **.env.dev.example** as reference
	- ENV: "dev"
	- LANGFUSE_SECRET_KEY: LangFuse Secret Key
	- LANGFUSE_PUBLIC_KEY: LangFuse Public Key
	- LANGFUSE_HOST: https://us.cloud.langfuse.com or self-hosted
	- GOOGLE_API_KEY: Google API Key
	- LLM_PROVIDER: "google" supported
	- LLM_MODEL: "gemini-2.0-flash" (ChatGoogleGenerativeAI available models)
5. Run the FastAPI application in development mode:
   ```bash
   fastapi dev main.py
   ```
6. Access:
	- Ingest endpoint: http://localhost:8000/ingest
	- Query endpoint: http://localhost:8000/query

### Production (Docker) Deployment
1. Set environment variables in **.env.prod**. You may use **.env.prod.example** as reference
	- ENV: "prod"
	- LANGFUSE_SECRET_KEY: LangFuse Secret Key
	- LANGFUSE_PUBLIC_KEY: LangFuse Public Key
	- LANGFUSE_HOST: https://us.cloud.langfuse.com (or self-hosted url)
	- GOOGLE_API_KEY: Google API Key
	- LLM_PROVIDER: "google" only supported currently
	- LLM_MODEL: "gemini-2.0-flash" (ChatGoogleGenerativeAI available models)
	- CHROMA_HOST: "chroma" (Container running ChromaDB)
	- CHROMA_PORT: 8000 (ChromaDB port)

2. Set additional Prometheus environment variables in **prometheus.env**. You may use **prometheus.env.example** as reference
	- PROMETHEUS_TARGET: localhost
	- RAG_API_TARGET: FastAPI container (rag-api)
	
2. Set additional Grafana environment variables in **grafana.env**. You may use **grafana.env.example** as reference
	- GF_SECURITY_ADMIN_PASSWORD: Set a password to access Grafana
	- GF_SECURITY_ADMIN_USER: Set an admin username
	- PROMETHEUS_HOST: Prometheus host (prometheus)
 
4. Build and run services in background using Docker Compose:
   ```bash
   docker-compose up --build -d
   ```
5. View logs
	 ```
	 docker-compose logs -f rag-api chroma
	 ```
6. Access:
	- Ingest endpoint: http://localhost:8000/ingest
	- Query endpoint: http://localhost:8000/query
	- Prometheus: http://localhost:9090
	- Grafana: http://localhost:3000


## Examples
Use the following cURL commands to ingest some documents:
```bash
curl -X POST "http://localhost:8000/ingest"   -H "Content-Type: application/json"   -d '{
"url": "https://fastapi.tiangolo.com/tutorial/handling-errors/",
"document_type": "html"
}'
```
```bash
curl -X POST "http://localhost:8000/ingest"   -H "Content-Type: application/json"   -d '{
"url": "https://python.langchain.com/docs/tutorials/rag/",
"document_type": "html"
}'
```
Once documents are ingested, you can query using cURL:
```bash
curl -X POST "http://localhost:8000/query"   -H "Content-Type: application/json"   -d '{
"question": "How do you use LangGraph to make a RAG application?"
}'
```

# Project File Tree
```
.
├── .dockerignore
├── .env.dev.example
├── .env.prod.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── README.md
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── lifespan_setup.py
│   │   ├── main.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── health.py
│   │       ├── ingest.py
│   │       ├── metrics.py
│   │       └── query.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── pydantic_settings.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── logging.py
│   │   ├── metrics.py
│   │   ├── chat_model/
│   │   │   ├── __init__.py
│   │   │   ├── llm.py
│   │   │   ├── prompt.py
│   │   │   └── services/
│   │   │       ├── __init__.py
│   │   │       └── external_llm.py
│   │   ├── embeddings/
│   │   │   ├── __init__.py
│   │   │   ├── compute_embeddings.py
│   │   │   └── embeddings_model.py
│   │   ├── exceptions/
│   │   │   ├── __init__.py
│   │   │   ├── exceptions.py
│   │   │   └── http_exceptions.py
│   │   ├── ingest/
│   │   │   ├── __init__.py
│   │   │   └── ingest.py
│   │   ├── langgraph/
│   │   │   ├── __init__.py
│   │   │   ├── langgraph.py
│   │   │   └── models.py
│   │   ├── loader/
│   │   │   ├── __init__.py
│   │   │   ├── content_loader.py
│   │   │   ├── text_splitter.py
│   │   │   └── url_loader.py
│   │   └── observability/
│   │       ├── __init__.py
│   │       └── langfuse.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py
│   └── services/
│       ├── __init__.py
│       └── vectorstore.py
├── grafana/
│   ├── dashboards/
│   │   └── app-monitoring.json
│   └── provisioning/
│       ├── dashboards/
│       │   └── dashboard.yml
│       └── datasources/
│           └── prometheus.yml
├── prometheus/
│   └── prometheus.yml
├── pyproject.toml
├── settings.py
└── test_main.py
```
# FastAPI App Structure

## Root Level
- **app/main.py**: The main FastAPI application file containing the FastAPI app instance and lifespan function for preloading base documents (PEP8 and Think Python). This file includes router configuration, metrics setup, and is the entry point for uvicorn or FastAPI dev.

## API Module
- **app/api/**: Contains all HTTP API-related components
- **app/api/lifespan_setup.py**: Handles document ingestion during FastAPI's lifespan initialization, similar to the ingest endpoint but with different logging. Documents to preload can be added here. 
- **app/api/route.py**: Aggregates all routes from the routes directory
- **app/api/routes/**: Contains individual endpoint implementations
  - **app/api/routes/health.py**: Health check endpoint returning a status dictionary
  - **app/api/routes/ingest.py**: Ingestion endpoint that calls core functions with idempotence checks
  - **app/api/routes/metrics.py**: Prometheus metrics endpoint
  - **app/api/routes/query.py**: Query endpoint that processes requests through the RAG application

## Configuration
- **app/config/**: Contains Pydantic settings management
- **app/config/pydantic_settings.py**: Handles environment variable configuration for the FastAPI app

## Core Application Logic
- **app/core/**: Contains RAG application business logic meant to be reusable across different interfaces
- **app/core/logging.py**: Configures structlog logger with production and development presets
- **app/core/metrics.py**: Sets up Prometheus middleware and metrics collection

### Chat Model Components
- **app/core/chat_model/**: Manages LLM interactions
- **app/core/chat_model/llm.py**: Initializes chat model instances
- **app/core/chat_model/prompt.py**: Defines LangChain prompt templates. 

### Embeddings Components
- **app/core/embeddings/**: Handles embedding model operations
- **app/core/embeddings/compute_embeddings.py**: Manages embedding calculation during document ingestion
- **app/core/embeddings/embeddings_model.py**: Initializes and exports the embeddings model instance

### Ingestion Components
- **app/core/ingest/ingest.py**: Provides ingestion functionality to other services with duplicate detection

### LangGraph Components
- **app/core/langgraph/langgraph.py**: Defines the RAG application graph with retrieve and generate nodes
- **app/core/langgraph/models.py**: Defines TypedDict state for LangGraph

### Document Loading Components
- **app/core/loader/context_loader.py**: Loads documents from content fields with checksum calculation
- **app/core/loader/text_splitter.py**: Implements recursive character text splitting
- **app/core/loader/url_loader.py**: Handles document loading from URLs with experimental HTML support

### Observability
- **app/core/observability/langfuse.py**: Manages Langfuse callback handler and object instances

### Vector Store
- **app/core/vector_store/vectorstore.py**: Chroma-based vector store implementation with in-memory testing and Docker production options

## Exceptions
- **app/exceptions/**: Contains exception definitions
- **app/exceptions/exceptions.py**: General application exceptions including duplicate document handling
- **app/exceptions/http_exceptions.py**: FastAPI HTTP exception subclasses
- **app/exceptions/handlers.py**: FastAPI exception handlers (currently unused)

## Models
- **app/models/schemas.py**: Pydantic BaseModels for requests, responses, and source data with validation for URL/content fields