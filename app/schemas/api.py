from pydantic import BaseModel, Field, constr
from typing import List, Optional, Dict, Any


class UploadResponse(BaseModel):
    """Response model for document upload endpoint"""
    document_ids: List[str] = Field(..., description="List of IDs for the uploaded documents")
    message: str = Field(..., description="Status message")


class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    query: constr(min_length=1) = Field(..., description="Natural language query to search documents")
    top_k: int = Field(5, description="Number of most relevant chunks to retrieve")


class SourceMetadata(BaseModel):
    """Model for source document metadata"""
    document_id: str = Field(..., description="ID of the source document")
    title: Optional[str] = Field(None, description="Title of the document")
    author: Optional[str] = Field(None, description="Author of the document")
    page_number: Optional[int] = Field(None, description="Page number in the document")
    chunk_id: str = Field(..., description="ID of the specific chunk")
    score: float = Field(..., description="Relevance score of the chunk")


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    answer: str = Field(..., description="Generated answer from the LLM")
    sources: List[SourceMetadata] = Field(..., description="Source documents used for the answer")
    success: bool = Field(True, description="Whether the query was successful")
    error_message: Optional[str] = Field(None, description="Error message if any")
