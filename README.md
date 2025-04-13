# Semantic Q/A Based Search Engine API

A RESTful API for semantic document search and question answering. This system allows users to upload PDF documents, which are processed, indexed, and made available for natural language queries. The API uses vector embeddings to find relevant document chunks and a Large Language Model (LLM) to generate answers based on the retrieved content.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Environment Variables](#environment-variables)
- [Installation](#installation)
- [API Documentation](#api-documentation)
- [Usage Examples](#usage-examples)
- [Testing](#testing)
- [Docker Deployment](#docker-deployment)
- [Troubleshooting](#troubleshooting)

## Features

- **Document Upload**: Accepts PDF files for processing and indexing
- **Text Extraction**: Extracts text content from PDFs
- **Chunking**: Splits text into overlapping chunks for better context preservation
- **Vector Embeddings**: Generates embeddings using OpenAI's models or Hugging Face alternatives
- **Semantic Search**: Performs similarity search using Qdrant vector database
- **Question Answering**: Uses a Large Language Model to generate answers based on retrieved document chunks
- **Metadata Tracking**: Preserves document metadata (title, author, page numbers) throughout the process

## Architecture

The system consists of the following components:

1. **FastAPI Web Server**: Provides the REST API endpoints
2. **Document Processor**: Extracts text from PDFs and splits it into chunks
3. **Embedding Service**: Generates vector embeddings for text chunks and queries
4. **Vector Database Service**: Stores and retrieves document chunks in Qdrant
5. **LLM Service**: Generates natural language answers based on retrieved chunks


## Prerequisites

- Python 3.8+
- [Qdrant](https://qdrant.tech/) (vector database)
- OpenAI API key (for embeddings and LLM)


## Environment Variables

Create a `.env` file in the project root directory based on the provided `.env.example`:

```
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# OpenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
EMBEDDING_MODEL=text-embedding-ada-002
LLM_MODEL=gpt-4

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=document_chunks
VECTOR_SIZE=1536

# Document Processing
CHUNK_SIZE=750
CHUNK_OVERLAP=150
```


## Installation

1. Clone the repository:

```bash
git clone git@github.com:Mohammed-Alhuraibi/Cybernetics-Lab-Assignment.git
cd Cybernetics-Lab-Assignment
```

2. Create and activate a virtual environment:

```bash
python -m venv venv       # If using python3 `python3 -m venv venv`
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up environment variables (see [Environment Variables](#environment-variables) section)

5. Run Qdrant (if not using a cloud instance):

```bash
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant
```

6. Start the API server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Documentation

After starting the server, access the interactive API documentation at:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### Endpoints

#### `POST /api/v1/upload`

Upload one or multiple PDF documents for processing and indexing.

**Parameters:**
- `files`: One or more PDF files (multipart/form-data)

**Response:**
```json
{
  "document_ids": ["d290f1ee-6c54-4b01-90e6-d701748f0851"],
  "message": "Successfully processed 1 documents"
}
```

#### `POST /api/v1/query`

Submit a natural language query to search through indexed documents.

**Request Body:**
```json
{
  "query": "What is the capital of France?",
  "top_k": 5
}
```

**Response:**
```json
{
  "answer": "The capital of France is Paris.",
  "sources": [
    {
      "document_id": "d290f1ee-6c54-4b01-90e6-d701748f0851",
      "title": "World Geography",
      "author": "John Smith",
      "page_number": 42,
      "chunk_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
      "score": 0.95
    }
  ],
  "success": true,
  "error_message": null
}
```

## Usage Examples

### Python Client Example

```python
import requests
import os

# Upload a document
def upload_document(file_path):
    url = "http://localhost:8000/api/v1/upload"
    with open(file_path, "rb") as f:
        # Use os.path.basename for platform-independent file name extraction
        files = {"files": (os.path.basename(file_path), f, "application/pdf")}
        response = requests.post(url, files=files)
    return response.json()

# Query for information
def query_documents(query, top_k=5):
    url = "http://localhost:8000/api/v1/query"
    payload = {"query": query, "top_k": top_k}
    response = requests.post(url, json=payload)
    return response.json()

# Main script
# Prompt user for the document path
document_path = input("Enter the full path to the PDF document you want to upload: ")

# Check if the file exists
if not os.path.exists(document_path):
    print("File not found. Please check the path and try again.")
    exit()

# Upload the document with error handling
try:
    document_response = upload_document(document_path)
    print(f"Uploaded document: {document_response}")
except Exception as e:
    print(f"Error uploading document: {e}")
    exit()

# Prompt user for the query
query = input("Enter your question about the document: ")

# Query the document with error handling
try:
    query_response = query_documents(query)
    print(f"Answer: {query_response['answer']}")
    print(f"Sources: {query_response['sources']}")
except Exception as e:
    print(f"Error querying document: {e}")

```

### cURL Examples

**Upload a document:**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/upload' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'files=@document.pdf'
```

**Query documents:**
```bash
curl -X 'POST' \
  'http://localhost:8000/api/v1/query' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "What is the main topic of the document?",
  "top_k": 5
}'
```

## Testing

The project includes unit and integration tests using pytest. To run the tests:

```bash
pytest
```

For test coverage reporting:

```bash
pytest --cov=app tests/
```

## Docker Deployment

A Dockerfile is provided for containerizing the application:

1. Build the Docker image:

```bash
docker build -t semantic-qa-api .
```

2. Run the container:

```bash
docker run -p 8000:8000 --env-file .env semantic-qa-api
```

3. For Docker Compose deployment with Qdrant:
> **Note:** Before running `docker compose up -d`, make sure no previous Docker containers are running to avoid conflicts (e.g., port collisions). You can check for running containers with `docker ps` and stop them using `docker stop <container_id>`.
```bash
docker compose up -d
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Issues**: Ensure your API key is valid and properly set in the environment variables.

2. **PDF Parsing Errors**: Some PDFs may be password-protected, corrupted, or use unsupported features. Try converting the PDF to a more standard format.

3. **Qdrant Connection Issues**: Verify that Qdrant is running and accessible at the configured host and port.

4. **Performance Concerns**: For large documents, consider adjusting the chunk size and overlap parameters in the environment variables.

For more detailed troubleshooting, check the application logs.
