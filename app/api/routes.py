import os
import shutil
import logging
import tempfile
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.schemas.api import UploadResponse, QueryRequest, QueryResponse, SourceMetadata
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.vector_db import VectorDBService
from app.services.llm_service import LLMService

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
document_processor = DocumentProcessor()
embedding_service = EmbeddingService()
vector_db = VectorDBService()
llm_service = LLMService()


@router.post("/upload", response_model=UploadResponse)
async def upload_document(files: List[UploadFile] = File(...)):
    """
    Upload and process PDF documents.
    
    Extracts text, chunks it, generates embeddings, and stores in vector DB.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    
    # Process each uploaded file
    document_ids = []
    try:
        for file in files:
            if not file.filename.endswith('.pdf'):
                continue  # Skip non-PDF files
            
            # Save the uploaded file to a temporary location
            temp_dir = tempfile.mkdtemp()
            temp_file_path = os.path.join(temp_dir, file.filename)
            
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process the PDF
            document_id, chunks = await document_processor.process_pdf(temp_file_path)
            
            # Generate embeddings for chunks
            texts = [chunk.text for chunk in chunks]
            embeddings = await embedding_service.get_embeddings(texts)
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                if i < len(embeddings):
                    chunk.embedding = embeddings[i]
            
            # Store chunks in vector database
            await vector_db.insert_chunks(chunks)
            
            # Add document ID to list
            document_ids.append(document_id)
            
            # Clean up temporary files
            shutil.rmtree(temp_dir)
        
        return UploadResponse(
            document_ids=document_ids,
            message=f"Successfully processed {len(document_ids)} documents"
        )
        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process documents: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_documents(query_request: QueryRequest):
    """
    Query documents using natural language.
    
    Embeds query, searches vector DB, and generates an answer using LLM.
    """
    try:
        # Validate query
        if not query_request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Generate embedding for query
        query_embedding = await embedding_service.get_embedding(query_request.query)
        
        # Search for relevant chunks
        search_results = await vector_db.search(
            query_vector=query_embedding,
            top_k=query_request.top_k
        )
        
        # Handle case where no results are found
        if not search_results:
            return QueryResponse(
                answer="I couldn't find any relevant information to answer your question.",
                sources=[],
                success=True
            )
        
        # Generate answer using LLM
        answer = await llm_service.generate_answer(query_request.query, search_results)
        
        # Prepare source metadata
        sources = []
        for result in search_results:
            chunk = result.chunk
            meta = chunk.metadata
            sources.append(
                SourceMetadata(
                    document_id=meta.get("document_id", ""),
                    title=meta.get("title", ""),
                    author=meta.get("author", ""),
                    page_number=meta.get("page_number"),
                    chunk_id=chunk.id,
                    score=result.score
                )
            )
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            success=True
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return QueryResponse(
            answer="An error occurred while processing your query.",
            sources=[],
            success=False,
            error_message=str(e)
        )
