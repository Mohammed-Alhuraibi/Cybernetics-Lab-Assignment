import openai
import logging
from typing import List, Dict, Any
from openai import OpenAI

from app.core.config import settings
from app.models.document import SearchResult

logger = logging.getLogger(__name__)

class LLMService:
    """Service for generating answers using OpenAI's LLM models"""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.LLM_MODEL
    
    async def generate_answer(self, query: str, search_results: List[SearchResult]) -> str:
        """
        Generate an answer using an LLM based on retrieved document chunks.
        
        Args:
            query: User's query
            search_results: List of retrieved document chunks
            
        Returns:
            Generated answer
        """
        try:
            if not search_results:
                return "I couldn't find any relevant information to answer your question."
            
            # Create context from search results
            context = self._create_context(search_results)
            
            # Create prompt for the LLM
            prompt = self._create_prompt(query, context)
            
            # Generate answer using OpenAI API (new format for v1.0+)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. If the answer cannot be found in the context, admit that you don't know rather than making up an answer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            # Extract answer from response
            answer = response.choices[0].message.content.strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise RuntimeError(f"Failed to generate answer: {str(e)}")
    
    def _create_context(self, search_results: List[SearchResult]) -> str:
        """
        Create context string from search results.
        
        Args:
            search_results: List of retrieved document chunks
            
        Returns:
            Context string
        """
        # Sort search results by score (highest first)
        sorted_results = sorted(search_results, key=lambda x: x.score, reverse=True)
        
        # Create context string
        context_parts = []
        for i, result in enumerate(sorted_results):
            chunk = result.chunk
            metadata = chunk.metadata
            
            # Format metadata
            meta_str = f"Document: {metadata.get('title', 'Untitled')}"
            if metadata.get('author'):
                meta_str += f", Author: {metadata['author']}"
            if metadata.get('page_number'):
                meta_str += f", Page: {metadata['page_number']}"
            
            # Add to context
            context_parts.append(f"[Source {i+1}: {meta_str}]\n{chunk.text}\n")
        
        return "\n".join(context_parts)
    
    def _create_prompt(self, query: str, context: str) -> str:
        """
        Create prompt for the LLM.
        
        Args:
            query: User's query
            context: Context string
            
        Returns:
            Prompt string
        """
        return f"""
Please answer the following question based ONLY on the provided context: make the answer short and concise.

Question: {query}

Context:
{context}

Answer:
"""
