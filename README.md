finished rewrite

cofigure grafana and prometheus using the template

and finally tests


# RAG API
This is a FastAPI RAG microservice application with endpoints to provide document URLs for ingestion and answer queries using an LLM with support for full observabilty using LangFuse. 
## Considerations:
Changes to the API: Since the example documents are both URLs, I figured it would make more sense to have an url field so the backend service doesn't have to interpret the string in the content field nor have to process potentially massive requests. I've left the content field for the sake of completeness but have added a constraint to requests so that either only the url or the content field are used (not both) and would like to note that I would probably remove the content field as is in a real scenario.

Tests can currently be run using the Development Environment instructions.

## Stack
- Python 3.13
- FastAPI
- LangChain / Langfuse (for RAG and observability)
- ChromaDB (for vector storage)
- LLM (Google GenAI)
- Docker, Docker Compose

## API Reference

### Health Check
- **GET** `/health` - Returns 200 OK

### Ingest Endpoint
**POST** `/ingest`
```json
{
  "url": "https://example.com",  // OR
  "content": "Raw text content",
  "document_type": "html|text|pdf|markdown"
}
```
returns:
```json
{
	"status": "success|error",
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
		{ "page": `<number>`, "text": "<passage text>" },
		…
	]
}
```
### Metrics Endpoint
- **GET** `/metrics` - Returns formatted metrics for Prometheus

## Dependencies
### Production Dependencies 
Refer to requirements.txt and docker-compose.yml

### Development Dependencies
Refer to requirements-dev.txt

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
   pip install -r requirements-dev.txt
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

7. Run tests:
   ```bash
   pytest test_main.py
   ```

### Production (Docker) Deployment
1. Set environment variables in **.env.prod**. You may use **.env.prod.example** as reference
	- ENV: "prod"
	- LANGFUSE_SECRET_KEY: LangFuse Secret Key
	- LANGFUSE_PUBLIC_KEY: LangFuse Public Key
	- LANGFUSE_HOST: https://us.cloud.langfuse.com or self-hosted
	- GOOGLE_API_KEY: Google API Key
	- LLM_PROVIDER: "google" only supported currently
	- LLM_MODEL: "gemini-2.0-flash" (ChatGoogleGenerativeAI available models)
	- CHROMA_HOST: "chroma" (Container running ChromaDB)
	- CHROMA_PORT: 8000 (ChromaDB port)

2. Set Grafana environment variables in **grafana.env**. You may use **grafana.env.example** as reference
	- GF_SECURITY_ADMIN_PASSWORD: Set a password to access Grafana
	- GF_SECURITY_ADMIN_USER: Set an admin username
 
3. Build and run the services using Docker Compose:
   ```bash
   docker-compose up --build
   ```
4. Access:
	- Ingest endpoint: http://localhost:8000/ingest
	- Query endpoint: http://localhost:8000/query
	- Prometheus: http://localhost:9090
	- Grafana: http://localhost:3000


## Examples
Use the following cURL commands to ingest some documents:
```bash
curl -X POST "http://localhost:8000/ingest"   -H "Content-Type: application/json"   -d '{
"url": "https://allendowney.github.io/ThinkPython/index.html",
"document_type": "html"
}'
```
```bash
curl -X POST "http://localhost:8000/ingest"   -H "Content-Type: application/json"   -d '{
"url": "https://peps.python.org/pep-0008/",
"document_type": "html"
}'
```
Once documents are ingested, you can query using cURL:
```bash
curl -X POST "http://localhost:8000/query"   -H "Content-Type: application/json"   -d '{
"question": "What should I do to program Python correctly?"
}'
```

## Project File Tree
```
.
├── .dockerignore
├── .env.dev.example
├── .env.prod.example
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── log_util.py
├── main.py
├── prometheus.yml
├── README.md
├── requirements-dev.txt
├── requirements.txt
├── settings.py
├── test_main.py
├── grafana.env.example
├── grafana/
│   ├── dashboards/
│   │   └── app-monitoring.json
│   └── provisioning/
│       ├── dashboards/
│       │   └── dashboard.yml
│       └── datasources/
│           └── prometheus.yml
