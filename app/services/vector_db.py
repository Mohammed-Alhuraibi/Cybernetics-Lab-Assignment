import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import settings
from app.models.document import DocumentChunk, SearchResult

logger = logging.getLogger(__name__)

class VectorDBService:
    """Service for interacting with Qdrant vector database"""
    
    def __init__(self):
        # Connect to Qdrant
        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT
        )
        self.collection_name = settings.COLLECTION_NAME
        self.vector_size = settings.VECTOR_SIZE
        
        # Initialize collection if it doesn't exist
        self._initialize_collection()
    
    def _initialize_collection(self):
        """Initialize Qdrant collection if it doesn't exist"""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            # Create new collection
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            logger.info(f"Created new collection: {self.collection_name}")
    
    async def insert_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Insert document chunks into the vector database.
        
        Args:
            chunks: List of DocumentChunk objects with embeddings
            
        Returns:
            Boolean indicating success
        """
        try:
            # Prepare points for insertion
            points = []
            for chunk in chunks:
                if not chunk.embedding:
                    logger.warning(f"Skipping chunk {chunk.id} - no embedding")
                    continue
                
                points.append(
                    models.PointStruct(
                        id=chunk.id,
                        vector=chunk.embedding,
                        payload={
                            "text": chunk.text,
                            "metadata": chunk.metadata
                        }
                    )
                )
            
            if points:
                # Insert points into collection
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"Inserted {len(points)} chunks into vector database")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error inserting chunks into vector database: {str(e)}")
            raise RuntimeError(f"Failed to insert chunks: {str(e)}")
    
    async def search(self, query_vector: List[float], top_k: int = 5) -> List[SearchResult]:
        """
        Search for the most similar document chunks.
        
        Args:
            query_vector: Embedding vector of the query
            top_k: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Search for similar vectors
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            
            # Convert to SearchResult objects
            results = []
            for res in search_results:
                chunk = DocumentChunk(
                    id=res.id,
                    text=res.payload["text"],
                    metadata=res.payload["metadata"]
                )
                results.append(SearchResult(chunk=chunk, score=res.score))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector database: {str(e)}")
            raise RuntimeError(f"Failed to search: {str(e)}")
    
    async def delete_by_document_id(self, document_id: str) -> bool:
        """
        Delete all chunks associated with a document ID.
        
        Args:
            document_id: ID of the document to delete
            
        Returns:
            Boolean indicating success
        """
        try:
            # Delete points with matching document_id in metadata
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="metadata.document_id",
                                match=models.MatchValue(value=document_id)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Deleted chunks for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document chunks: {str(e)}")
            raise RuntimeError(f"Failed to delete document: {str(e)}")
