import os
import re
import uuid
import tempfile
from typing import List, Dict, Any, Optional, Generator, Tuple
import fitz  # PyMuPDF
from nltk.tokenize import sent_tokenize
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PDFProcessor")

# Import nltk and download necessary data
try:
    import nltk
    nltk.download('punkt', quiet=True)
except ImportError:
    logger.warning("NLTK not installed. Using basic sentence splitting.")
    def sent_tokenize(text):
        return re.split(r'(?<=[.!?])\s+', text)

class PDFProcessor:
    """
    Process PDF documents into chunks for vectorization.
    
    This class:
    1. Reads PDF files from a directory
    2. Extracts text and metadata
    3. Chunks the text into smaller pieces with appropriate overlap
    4. Adds metadata to each chunk to enable proper sourcing and citation
    """
    
    def __init__(self, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200,
                 semantic_chunking: bool = True,
                 max_pages_in_memory: int = 5,
                 page_buffer_size: int = 10):
        """
        Initialize the PDF processor.
        
        Args:
            chunk_size: Maximum size of document chunks in characters
            chunk_overlap: Overlap between chunks in characters
            semantic_chunking: Whether to use semantic chunking (preserve sentences)
            max_pages_in_memory: Maximum number of pages to process in memory at once
            page_buffer_size: Number of pages to buffer when streaming large documents
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.semantic_chunking = semantic_chunking
        self.max_pages_in_memory = max_pages_in_memory
        self.page_buffer_size = page_buffer_size
        
        # Initialize cache for chunking results
        self._chunk_cache = {}
        self._cache_size_limit = 100  # Maximum number of files to cache
    
    def process_pdf_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all PDF files in a directory.
        
        Args:
            directory_path: Path to directory containing PDF files
            
        Returns:
            List of dictionaries with 'id', 'text', and 'metadata' for each chunk
        """
        all_chunks = []
        
        # Find all PDF files in the directory
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        total_files = len(pdf_files)
        
        logger.info(f"Found {total_files} PDF files in {directory_path}")
        
        # Process each PDF file
        for i, pdf_file in enumerate(pdf_files):
            file_path = os.path.join(directory_path, pdf_file)
            logger.info(f"Processing file {i+1}/{total_files}: {pdf_file}")
            
            # Check if file is in cache
            file_stats = os.stat(file_path)
            file_key = f"{file_path}_{file_stats.st_mtime}_{file_stats.st_size}"
            
            if file_key in self._chunk_cache:
                logger.info(f"Using cached chunks for {pdf_file}")
                file_chunks = self._chunk_cache[file_key]
            else:
                # Process the file
                file_chunks = self.process_pdf_file(file_path)
                
                # Update cache (with LRU policy)
                if len(self._chunk_cache) >= self._cache_size_limit:
                    # Remove oldest item
                    oldest_key = next(iter(self._chunk_cache))
                    del self._chunk_cache[oldest_key]
                
                # Add to cache
                self._chunk_cache[file_key] = file_chunks
                
            all_chunks.extend(file_chunks)
            
            # Log progress
            logger.info(f"Added {len(file_chunks)} chunks from {pdf_file}")
        
        return all_chunks
    
    def process_pdf_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a single PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of dictionaries with 'id', 'text', and 'metadata' for each chunk
        """
        try:
            # Open the PDF file
            document = fitz.open(file_path)
            
            # Extract document metadata
            file_metadata = {
                "source": os.path.basename(file_path),
                "file_path": file_path,
                "page_count": len(document),
                "title": document.metadata.get("title", ""),
                "author": document.metadata.get("author", ""),
                "subject": document.metadata.get("subject", ""),
                "creation_date": document.metadata.get("creationDate", "")
            }
            
            # Check if the document is too large for memory processing
            if len(document) > self.max_pages_in_memory:
                logger.info(f"Large document detected ({len(document)} pages). Using streaming processing.")
                chunks = self._process_large_pdf(document, file_metadata)
            else:
                # Process normally for smaller documents
                chunks = self._process_standard_pdf(document, file_metadata)
            
            # Close the document
            document.close()
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            return []
    
    def _process_standard_pdf(self, document, file_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a standard-sized PDF document."""
        chunks = []
        
        # Process each page
        for page_num, page in enumerate(document):
            # Extract text from page
            text = page.get_text()
            
            # Skip if text is empty
            if not text.strip():
                continue
            
            # Chunk the text
            if self.semantic_chunking:
                page_chunks = self._create_semantic_chunks(text)
            else:
                page_chunks = self._create_chunks(text)
            
            # Add metadata to each chunk
            for i, chunk_text in enumerate(page_chunks):
                chunk = self._create_chunk_with_metadata(
                    chunk_text, file_metadata, page_num, i, len(page_chunks)
                )
                chunks.append(chunk)
        
        return chunks
    
    def _process_large_pdf(self, document, file_metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a large PDF document using streaming to reduce memory usage."""
        chunks = []
        total_pages = len(document)
        
        # Process pages in batches
        for start_page in range(0, total_pages, self.page_buffer_size):
            end_page = min(start_page + self.page_buffer_size, total_pages)
            batch_text = ""
            current_page_num = start_page
            
            # Accumulate text from the batch of pages
            for page_num in range(start_page, end_page):
                page = document[page_num]
                page_text = page.get_text()
                
                if not page_text.strip():
                    continue
                
                # If the accumulated text is getting large, process it
                if len(batch_text) + len(page_text) > self.chunk_size * 3:
                    # Process the accumulated text
                    batch_chunks = self._process_text_batch(batch_text, file_metadata, current_page_num)
                    chunks.extend(batch_chunks)
                    
                    # Reset the batch
                    batch_text = ""
                    current_page_num = page_num
                
                # Add the current page to the batch
                batch_text += page_text + "\n\n"
            
            # Process any remaining text
            if batch_text:
                batch_chunks = self._process_text_batch(batch_text, file_metadata, current_page_num)
                chunks.extend(batch_chunks)
            
            # Log progress
            logger.info(f"Processed pages {start_page+1}-{end_page} of {total_pages}")
        
        return chunks
    
    def _process_text_batch(self, text: str, file_metadata: Dict[str, Any], page_num: int) -> List[Dict[str, Any]]:
        """Process a batch of text from one or more pages."""
        chunks = []
        
        # Chunk the text
        if self.semantic_chunking:
            text_chunks = self._create_semantic_chunks(text)
        else:
            text_chunks = self._create_chunks(text)
        
        # Add metadata to each chunk
        for i, chunk_text in enumerate(text_chunks):
            chunk = self._create_chunk_with_metadata(
                chunk_text, file_metadata, page_num, i, len(text_chunks)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _create_chunk_with_metadata(self, chunk_text: str, file_metadata: Dict[str, Any], 
                                  page_num: int, chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata."""
        chunk_id = str(uuid.uuid4())
        
        # Create chunk metadata
        chunk_metadata = file_metadata.copy()
        chunk_metadata.update({
            "page_number": page_num + 1,
            "chunk_number": chunk_index + 1,
            "chunk_count": total_chunks,
            "content_preview": chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text
        })
        
        # Return the chunk
        return {
            "id": chunk_id,
            "text": chunk_text,
            "metadata": chunk_metadata
        }
    
    def _create_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks of approximately chunk_size characters with overlap.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Get chunk of specified size
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to find a good break point
            if end < len(text):
                # Try to find the last period or line break in the chunk
                last_period = text.rfind('.', start, end)
                last_newline = text.rfind('\n', start, end)
                
                # Use the later of period or newline if found within a reasonable range
                break_point = max(last_period, last_newline)
                if break_point > start + (self.chunk_size // 2):
                    end = break_point + 1  # Include the period or newline
            
            # Extract the chunk
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move start position for next chunk, considering overlap
            start = end - self.chunk_overlap if end < len(text) else len(text)
        
        return chunks
    
    def _create_semantic_chunks(self, text: str) -> List[str]:
        """
        Split text into chunks respecting sentence boundaries.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize into sentences
        sentences = sent_tokenize(text)
        
        # Initialize variables
        chunks = []
        current_chunk = ""
        
        # Process each sentence
        for sentence in sentences:
            # If adding this sentence would exceed chunk_size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Save the current chunk
                chunks.append(current_chunk.strip())
                
                # Start a new chunk with overlap
                overlap_size = min(self.chunk_overlap, len(current_chunk))
                if overlap_size > 0:
                    # Get the last few sentences for overlap
                    overlap_sentences = sent_tokenize(current_chunk[-overlap_size:])
                    if overlap_sentences:
                        current_chunk = " ".join(overlap_sentences[-2:]) + " "
                    else:
                        current_chunk = ""
                else:
                    current_chunk = ""
            
            # Add the sentence to the current chunk
            current_chunk += sentence + " "
        
        # Add the last chunk if not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks 