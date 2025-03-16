import os
import logging
from typing import List, Dict, Any, Optional
from llama_cloud_services import LlamaParse
from llama_index.core import SimpleDirectoryReader
from agno.document.base import Document as AgnoDocument
from src.utils.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PDFParser")

class PDFProcessor:
    """
    Process PDF documents into Agno Document objects using LlamaParse.
    
    This class:
    1. Reads PDF files from a directory or path
    2. Uses LlamaParse to extract text and metadata
    3. Creates Agno Document objects for further processing
    """
    
    def __init__(self):
        """Initialize the PDF processor with LlamaParse."""
        self.parser = LlamaParse(
            api_key=Config.LLAMA_CLOUD_API_KEY,
            result_type="markdown"  # "markdown" and "text" are available
        )
        
        # Initialize cache for parsing results
        self._parse_cache = {}
        self._cache_size_limit = 100  # Maximum number of files to cache
    
    def process_documents(self, path: str) -> Dict[str, Any]:
        """
        Process document file(s) from a path using LlamaCloud Parsing.
        Handles both single files and directories.
        
        Args:
            path: Path to document file or directory
            
        Returns:
            Dict containing status and results with AgnoDocuments
        """
        logger.info(f"Processing document path: {path}")
        
        # Check if path exists
        if not os.path.exists(path):
            logger.error(f"Path does not exist: {path}")
            return {
                "success": False,
                "error": f"Path {path} does not exist",
                "document_count": 0,
                "documents": []
            }

        # Get list of PDF files to process
        pdf_files = []
        if os.path.isfile(path):
            if path.lower().endswith('.pdf'):
                pdf_files = [path]
            else:
                return {
                    "success": False,
                    "error": "Only PDF files are supported",
                    "document_count": 0,
                    "documents": []
                }
        else:
            pdf_files = [f for f in os.listdir(path) if f.lower().endswith('.pdf')]
            pdf_files = [os.path.join(path, f) for f in pdf_files]
            
        if not pdf_files:
            logger.warning(f"No PDF files found in {path}")
            return {
                "success": False,
                "error": f"No PDF files found in {path}",
                "document_count": 0,
                "documents": []
            }

        # Process each PDF file
        processed_documents = []
        processed_files = []
        failed_files = []
        
        for pdf_file in pdf_files:
            try:
                # Check if file is in cache
                file_stats = os.stat(pdf_file)
                file_key = f"{pdf_file}_{file_stats.st_mtime}_{file_stats.st_size}"
                
                if file_key in self._parse_cache:
                    logger.info(f"Using cached document for {os.path.basename(pdf_file)}")
                    document = self._parse_cache[file_key]
                else:
                    # Process the file
                    document = self._parse_pdf(pdf_file)
                    
                    if document:
                        # Update cache (with LRU policy)
                        if len(self._parse_cache) >= self._cache_size_limit:
                            # Remove oldest item
                            oldest_key = next(iter(self._parse_cache))
                            del self._parse_cache[oldest_key]
                        
                        # Add to cache
                        self._parse_cache[file_key] = document
                
                if document:
                    processed_documents.append(document)
                    processed_files.append(os.path.basename(pdf_file))
                    logger.debug(f"Successfully processed {pdf_file}")
                
            except Exception as e:
                logger.error(f"Error processing file {pdf_file}: {str(e)}", exc_info=True)
                failed_files.append({
                    "file": os.path.basename(pdf_file),
                    "error": str(e)
                })

        # Return results
        if not processed_files:
            return {
                "success": False,
                "error": "Failed to process any files",
                "failed_files": failed_files,
                "document_count": 0,
                "documents": []
            }

        return {
            "success": True,
            "documents": processed_documents,
            "processed_files": processed_files,
            "failed_files": failed_files if failed_files else None,
            "document_count": len(processed_files),
            "details": f"Successfully processed {len(processed_files)} files"
        }
    
    def _parse_pdf(self, file_path: str) -> Optional[AgnoDocument]:
        """
        Parse a PDF file using LlamaParse and convert to Agno Document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Agno Document object if successful, None otherwise
        """
        logger.debug(f"Parsing PDF: {file_path}")
        
        # Parse PDF using LlamaParse
        llama_documents = SimpleDirectoryReader(
            input_files=[file_path],
            file_extractor={'.pdf': self.parser}
        ).load_data()
        
        # Get content as markdown
        if not llama_documents or len(llama_documents) == 0:
            logger.warning(f"No content extracted from {file_path}")
            return None
            
        markdown_content = "\n\n".join([doc.get_content() for doc in llama_documents])
        
        # Create Agno Document
        file_name = os.path.basename(file_path)
        document = AgnoDocument(
            id=file_name,
            name=file_name,
            content=markdown_content,
            meta_data={
                "source": file_name,
                "file_path": file_path,
                "file_type": "pdf"
            }
        )
        
        return document 