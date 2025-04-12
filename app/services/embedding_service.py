import openai
import logging
from typing import List, Union
from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Service for generating embeddings using OpenAI's API"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts using OpenAI's embedding model.
        
        Args:
            texts: List of text strings to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        try:
            if not texts:
                return []
            
            # Create embeddings using OpenAI API (new format for v1.0+)
            response = self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            
            # Extract embedding data from response
            embeddings = [data.embedding for data in response.data]
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise RuntimeError(f"Failed to generate embeddings: {str(e)}")
    
    async def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using OpenAI's embedding model.
        
        Args:
            text: Text string to generate embedding for
            
        Returns:
            Embedding vector
        """
        result = await self.get_embeddings([text])
        return result[0] if result else []
