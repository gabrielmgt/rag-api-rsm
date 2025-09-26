import pytest
from fastapi.testclient import TestClient
from main import app, IngestRequest, DocumentType, ChatGoogleGenerativeAI, langfuse_callback_handler
from pydantic import ValidationError, HttpUrl 
from unittest.mock import patch, Mock, AsyncMock
from langchain_core.documents import Document
from langchain_core.language_models.fake import FakeListLLM


# Test client for FastAPI application
@pytest.fixture
def test_client():
    return TestClient(app)

class MyFakeLLM(FakeListLLM):
    def __init__(self, responses):
        super().__init__(responses=responses)
    
    def _call(self, prompt, **kwargs):
        self._prompt = prompt  # Capture the input prompt
        return super()._call(prompt, **kwargs)
    
    def get_prompt(self):
        return self._prompt  # Retrieve the last prompt used

# Test /health endpoint
def test_health_endpoint(test_client: TestClient):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

# Test /metrics endpoint
def test_metrics_endpoint(test_client: TestClient):
    response = test_client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    assert "http_requests_total" in response.text
    assert "request_duration_seconds_bucket" in response.text
    assert "cpu_usage_percent" in response.text
    assert "memory_usage_percent" in response.text

# Test IngestRequest validation
def test_ingest_request_validation_both_provided():
    with pytest.raises(ValidationError) as exc_info:
        IngestRequest(content="some content", url=HttpUrl("http://example.com"), document_type=DocumentType.TEXT)
    assert "Provide either url or content, not both" in str(exc_info.value)

def test_ingest_request_validation_neither_provided():
    with pytest.raises(ValidationError) as exc_info:
        IngestRequest(document_type=DocumentType.TEXT)
    assert "Either url or content must be provided" in str(exc_info.value)

def test_ingest_request_validation_only_url():
    request = IngestRequest(url=HttpUrl("http://example.com"), document_type=DocumentType.HTML)
    assert str(request.url) == "http://example.com/" 
    assert request.content is None
    assert request.document_type == DocumentType.HTML

def test_ingest_request_validation_only_content():
    request = IngestRequest(content="some text", document_type=DocumentType.TEXT)
    assert request.content == "some text"
    assert request.url is None
    assert request.document_type == DocumentType.TEXT

# Test /ingest endpoint
@pytest.mark.asyncio
@patch('main.computate_embeddings_and_add_to_store', new_callable=AsyncMock)
@patch('main.split_documents_with_tracing', new_callable=Mock)
@patch('main.load_document_from_url', new_callable=AsyncMock)
@patch('main.load_document_from_content', new_callable=Mock)
async def test_ingest_document_success_url(
    mock_load_document_from_content,
    mock_load_document_from_url,
    mock_split_documents_with_tracing,
    mock_computate_embeddings_and_add_to_store,
    test_client: TestClient
):
    mock_docs = [Document(page_content="url content", metadata={"source": "test"})]
    mock_chunks = [Document(page_content="chunk 1", metadata={"source": "test"})]
    
    mock_load_document_from_url.return_value = mock_docs
    mock_split_documents_with_tracing.return_value = mock_chunks
    mock_computate_embeddings_and_add_to_store.return_value = None

    response = test_client.post(
        "/ingest",
        json={"url": "https://fastapi.tiangolo.com/tutorial/testing/", "document_type": "html"}
    )
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["chunks_created"] == len(mock_chunks)
    mock_load_document_from_url.assert_called_once()
    mock_split_documents_with_tracing.assert_called_once_with(mock_docs)
    mock_computate_embeddings_and_add_to_store.assert_called_once_with(mock_chunks)
    mock_load_document_from_content.assert_not_called()

@pytest.mark.asyncio
@patch('main.computate_embeddings_and_add_to_store', new_callable=AsyncMock)
@patch('main.split_documents_with_tracing', new_callable=Mock)
@patch('main.load_document_from_url', new_callable=AsyncMock)
@patch('main.load_document_from_content', new_callable=Mock)
async def test_ingest_document_success_content(
    mock_load_document_from_content,
    mock_load_document_from_url,
    mock_split_documents_with_tracing,
    mock_computate_embeddings_and_add_to_store,
    test_client: TestClient
):
    mock_docs = [Document(page_content="some content", metadata={"source": "test"})]
    mock_chunks = [Document(page_content="chunk 1", metadata={"source": "test"})]
    
    mock_load_document_from_content.return_value = mock_docs
    mock_split_documents_with_tracing.return_value = mock_chunks
    mock_computate_embeddings_and_add_to_store.return_value = None

    response = test_client.post(
        "/ingest",
        json={"content": "This is some test content.", "document_type": "text"}
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["chunks_created"] == len(mock_chunks)
    mock_load_document_from_content.assert_called_once()
    mock_split_documents_with_tracing.assert_called_once_with(mock_docs)
    mock_computate_embeddings_and_add_to_store.assert_called_once_with(mock_chunks)
    mock_load_document_from_url.assert_not_called()

@pytest.mark.asyncio
@patch('main.vectorstore.aadd_documents', new_callable=Mock)
@patch('main.load_document_from_url', new_callable=Mock)
async def test_ingest_document_failure(
    mock_load_document_from_url,
    mock_aadd_documents,
    test_client: TestClient
):
    mock_load_document_from_url.side_effect = Exception("Failed to load document")

    response = test_client.post(
        "/ingest",
        json={"url": "http://bad.url", "document_type": "html"}
    )

    assert response.status_code == 200 # FastAPI returns 200 for custom error responses
    assert response.json()["status"] == "error"
    assert "Failed to load document" in response.json()["message"]
    assert response.json()["chunks_created"] == 0
    mock_load_document_from_url.assert_called_once()
    mock_aadd_documents.assert_not_called()

@pytest.mark.asyncio
@patch('main.vectorstore.as_retriever')
@patch(
        "main.chat_model",
        return_value=MyFakeLLM(
            responses=["AI stands for Artificial Intelligence"],
        ),
    )
async def test_query_document_with_mocked_llm_response(
    mock_chat_model,
    mock_as_retriever,
    test_client: TestClient
):
    mock_retriever = AsyncMock()
    mock_as_retriever.return_value = mock_retriever
    
    mock_docs = [
        Document(page_content="This is a test document about AI.", metadata={"page": 1}),
        Document(page_content="AI stands for Artificial Intelligence.", metadata={"page": 2})
    ]
    mock_retriever.ainvoke.return_value = mock_docs
    
    response = test_client.post(
        "/query",
        json={"question": "What does AI stand for?"}
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "AI stands for Artificial Intelligence"
    assert len(response.json()["sources"]) == 2
    assert response.json()["sources"][0]["text"] == "This is a test document about AI."
    assert response.json()["sources"][1]["text"] == "AI stands for Artificial Intelligence."
    
    mock_as_retriever.assert_called_once()
    mock_retriever.ainvoke.assert_called_once_with(
        "What does AI stand for?",
        config={"callbacks": [langfuse_callback_handler]}
    )
    