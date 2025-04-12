import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_db import VectorDBService
from app.services.llm_service import LLMService
from app.models.document import DocumentChunk, SearchResult


# Create test client
client = TestClient(app)


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "docs_url" in response.json()


@patch.object(DocumentProcessor, "process_pdf")
@patch.object(EmbeddingService, "get_embeddings")
@patch.object(VectorDBService, "insert_chunks")
def test_upload_endpoint(mock_insert_chunks, mock_get_embeddings, mock_process_pdf):
    """Test the upload endpoint with mocked dependencies."""
    # Set up mock returns
    document_id = "test-doc-123"
    chunks = [
        DocumentChunk(
            id="chunk-1",
            text="Test chunk content",
            metadata={"document_id": document_id, "title": "Test Doc", "page_number": 1}
        )
    ]

    mock_process_pdf.return_value = (document_id, chunks)
    mock_get_embeddings.return_value = [[0.1, 0.2, 0.3]]
    mock_insert_chunks.return_value = True

    # Create a test PDF file
    test_file_path = "test_document.pdf"
    with open(test_file_path, "wb") as f:
        f.write(b"%PDF-1.5\n%Test PDF file")

    try:
        # Send the test request
        with open(test_file_path, "rb") as f:
            response = client.post(
                "/api/v1/upload",
                files={"files": ("test_document.pdf", f, "application/pdf")}
            )

        # Check response
        assert response.status_code == 200
        assert "document_ids" in response.json()
        assert document_id in response.json()["document_ids"]

        # Verify our mocks were called
        mock_process_pdf.assert_called_once()
        mock_get_embeddings.assert_called_once()
        mock_insert_chunks.assert_called_once()

    finally:
        # Clean up test file
        if os.path.exists(test_file_path):
            os.remove(test_file_path)


@patch.object(EmbeddingService, "get_embedding")
@patch.object(VectorDBService, "search")
@patch.object(LLMService, "generate_answer")
def test_query_endpoint(mock_generate_answer, mock_search, mock_get_embedding):
    """Test the query endpoint with mocked dependencies."""
    # Set up mock returns
    query = "What is the test about?"
    mock_get_embedding.return_value = [0.1, 0.2, 0.3]

    chunk = DocumentChunk(
        id="chunk-1",
        text="This test is about testing the API.",
        metadata={"document_id": "test-doc", "title": "Test Doc", "page_number": 1}
    )
    search_results = [SearchResult(chunk=chunk, score=0.95)]

    mock_search.return_value = search_results
    mock_generate_answer.return_value = "The test is about testing the API."

    # Send test request
    response = client.post(
        "/api/v1/query",
        json={"query": query, "top_k": 5}
    )

    # Check response
    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == "The test is about testing the API."
    assert result["success"] is True
    assert len(result["sources"]) == 1
    assert result["sources"][0]["document_id"] == "test-doc"

    # Verify our mocks were called
    mock_get_embedding.assert_called_once_with(query)
    mock_search.assert_called_once()
    mock_generate_answer.assert_called_once()


def test_empty_query():
    """Test the query endpoint with an empty query."""
    response = client.post(
        "/api/v1/query",
        json={"query": "", "top_k": 5}
    )

    assert response.status_code == 422


@patch.object(EmbeddingService, "get_embedding")
@patch.object(VectorDBService, "search")
def test_query_no_results(mock_search, mock_get_embedding):
    """Test the query endpoint when no results are found."""
    # Set up mock returns
    mock_get_embedding.return_value = [0.1, 0.2, 0.3]
    mock_search.return_value = []

    # Send test request
    response = client.post(
        "/api/v1/query",
        json={"query": "Unknown query", "top_k": 5}
    )

    # Check response
    assert response.status_code == 200
    result = response.json()
    assert "I couldn't find any relevant information" in result["answer"]
    assert result["success"] is True
    assert len(result["sources"]) == 0
