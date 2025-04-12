import fitz  # PyMuPDF
import uuid
import os
from typing import List, Dict, Any, Tuple, Optional
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import settings
from app.models.document import DocumentChunk

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Service for processing PDF documents"""
    
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
    
    def extract_text_from_pdf(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and metadata from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple containing extracted text and metadata
        """
        try:
            # Open the PDF file
            doc = fitz.open(file_path)
            
            # Extract metadata
            metadata = {
                "title": doc.metadata.get("title", os.path.basename(file_path)),
                "author": doc.metadata.get("author", "Unknown"),
                "total_pages": len(doc)
            }
            
            # Extract text from each page
            full_text = ""
            for page_num, page in enumerate(doc):
                text = page.get_text()
                full_text += f"Page {page_num + 1}:\n{text}\n\n"
            
            doc.close()
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise ValueError(f"Failed to process PDF: {str(e)}")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """
        Split text into chunks with overlapping segments.
        
        Args:
            text: Text to chunk
            metadata: Document metadata
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = self.text_splitter.split_text(text)
        document_chunks = []
        
        for i, chunk_text in enumerate(chunks):
            # Try to identify page number from the chunk
            page_number = None
            for line in chunk_text.split('\n'):
                if line.startswith("Page ") and ":" in line:
                    try:
                        page_number = int(line.split("Page ")[1].split(":")[0].strip())
                        break
                    except ValueError:
                        pass
            
            # Create DocumentChunk
            chunk_id = str(uuid.uuid4())
            document_chunks.append(
                DocumentChunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "document_id": metadata.get("document_id", ""),
                        "title": metadata.get("title", ""),
                        "author": metadata.get("author", ""),
                        "page_number": page_number,
                        "chunk_index": i
                    }
                )
            )
        
        return document_chunks
    
    async def process_pdf(self, file_path: str, document_id: Optional[str] = None) -> Tuple[str, List[DocumentChunk]]:
        """
        Process a PDF file: extract text, metadata, and create chunks.
        
        Args:
            file_path: Path to the PDF file
            document_id: Optional document ID (generated if not provided)
            
        Returns:
            Tuple containing document ID and list of document chunks
        """
        # Generate document ID if not provided
        if not document_id:
            document_id = str(uuid.uuid4())
        
        # Extract text and metadata
        text, metadata = self.extract_text_from_pdf(file_path)
        
        # Add document ID to metadata
        metadata["document_id"] = document_id
        
        # Chunk the text
        chunks = self.chunk_text(text, metadata)
        
        logger.info(f"Processed document {document_id}: {len(chunks)} chunks created")
        
        return document_id, chunks
