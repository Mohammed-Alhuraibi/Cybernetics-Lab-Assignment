from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class DocumentChunk(BaseModel):
    """Model representing a chunk of a document with its metadata"""
    id: str = Field(..., description="Unique identifier for the chunk")
    text: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(..., description="Metadata about the chunk and its source document")
    embedding: Optional[list] = Field(None, description="Vector embedding of the chunk text")


class SearchResult(BaseModel):
    """Model representing a search result from the vector database"""
    chunk: DocumentChunk = Field(..., description="The retrieved document chunk")
    score: float = Field(..., description="Similarity score for this chunk")
