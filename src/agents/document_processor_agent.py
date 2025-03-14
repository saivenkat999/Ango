import os
import glob
import logging
from typing import List, Dict, Any, Optional
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import Toolkit
from pydantic import BaseModel, Field
from src.utils.pdf_utils import PDFProcessor
from src.utils.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DocumentProcessorAgent")

class ProcessingResult(BaseModel):
    """Structured result from document processing operations."""
    success: bool = Field(description="Whether the operation was successful")
    document_count: int = Field(description="Number of documents processed")
    chunk_count: int = Field(description="Number of chunks created")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    details: Optional[str] = Field(None, description="Additional details about the operation")

class PDFProcessorToolkit(Toolkit):
    """Toolkit for processing PDF documents."""
    
    def __init__(self):
        super().__init__(name="pdf_processor")
        self.register(self.process_pdf_file)
        self.register(self.process_pdf_directory)
    
    def process_pdf_file(self, file_path: str, chunk_size: int = 512, chunk_overlap: int = 128) -> Dict[str, Any]:
        """
        Process a single PDF file and return the chunks.
        
        Args:
            file_path: Path to the PDF file
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            Dict containing status and results
        """
        logger.info(f"Processing PDF file: {file_path}")
        
        # Create the PDF processor
        processor = PDFProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_semantic_chunking=True
        )
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return {
                "success": False,
                "error": f"File {file_path} does not exist",
                "document_count": 0,
                "chunk_count": 0
            }
        
        try:
            # Process the PDF
            chunks = processor.process_pdf_file(file_path)
            logger.debug(f"Successfully processed {file_path}: {len(chunks)} chunks")
            
            return {
                "success": True,
                "chunks": chunks,
                "document_count": 1,
                "chunk_count": len(chunks),
                "details": f"Successfully processed file with {len(chunks)} chunks"
            }
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Error processing file {file_path}: {str(e)}",
                "document_count": 0,
                "chunk_count": 0
            }
    
    def process_pdf_directory(self, directory_path: str, chunk_size: int = 512, chunk_overlap: int = 128) -> Dict[str, Any]:
        """
        Process all PDF files in a directory and return the chunks.
        
        Args:
            directory_path: Path to the directory containing PDF files
            chunk_size: Size of each text chunk
            chunk_overlap: Overlap between consecutive chunks
            
        Returns:
            Dict containing status and results
        """
        logger.info(f"Processing PDF directory: {directory_path}")
        
        # Create the PDF processor
        processor = PDFProcessor(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_semantic_chunking=True
        )
        
        # Check if directory exists
        if not os.path.exists(directory_path):
            logger.error(f"Directory does not exist: {directory_path}")
            return {
                "success": False,
                "error": f"Directory {directory_path} does not exist",
                "document_count": 0,
                "chunk_count": 0
            }
        
        # Get list of PDF files
        pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
        logger.debug(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return {
                "success": False,
                "error": f"No PDF files found in {directory_path}",
                "document_count": 0,
                "chunk_count": 0
            }
        
        # Process each PDF file
        all_chunks = []
        processed_files = []
        failed_files = []
        
        for pdf_file in pdf_files:
            try:
                logger.debug(f"Processing file: {pdf_file}")
                # Process the PDF
                chunks = processor.process_pdf_file(pdf_file)
                
                # Add to results
                all_chunks.extend(chunks)
                processed_files.append(os.path.basename(pdf_file))
                logger.debug(f"Successfully processed {pdf_file}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"Error processing file {pdf_file}: {str(e)}", exc_info=True)
                failed_files.append({
                    "file": os.path.basename(pdf_file),
                    "error": str(e)
                })
        
        # Check if any files were processed successfully
        if not processed_files:
            return {
                "success": False,
                "error": "Failed to process any files",
                "failed_files": failed_files,
                "document_count": 0,
                "chunk_count": 0
            }
        
        # Return success with any partially successful results
        return {
            "success": True,
            "chunks": all_chunks,
            "processed_files": processed_files,
            "failed_files": failed_files if failed_files else None,
            "document_count": len(processed_files),
            "chunk_count": len(all_chunks),
            "details": f"Successfully processed {len(processed_files)} files with {len(all_chunks)} total chunks"
        }

class VectorStoreToolkit(Toolkit):
    """Toolkit for managing vector store operations."""
    
    def __init__(self, vector_store: VectorStore):
        super().__init__(name="vector_store_manager")
        self.vector_store = vector_store
        self.register(self.add_documents)
        self.register(self.get_collection_info)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document chunks to add
            
        Returns:
            Dict containing operation status and results
        """
        try:
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            return {
                "success": True,
                "document_count": len(documents),
                "details": f"Successfully added {len(documents)} chunks to vector store"
            }
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Error adding documents to vector store: {str(e)}",
                "document_count": 0
            }
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the vector store collection.
        
        Returns:
            Dict containing collection information
        """
        try:
            # Get count of items in collection
            collection_count = self.vector_store.collection.count()
            return {
                "success": True,
                "collection_name": self.vector_store.collection_name,
                "document_count": collection_count,
                "details": f"Collection '{self.vector_store.collection_name}' contains {collection_count} document chunks"
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Error getting collection info: {str(e)}"
            }

class DocumentProcessorAgent:
    """
    Agent responsible for processing documents and creating a knowledge base.
    
    This agent:
    1. Processes PDF documents from a specified directory
    2. Extracts and chunks text from the documents
    3. Creates and updates a ChromaDB vector store with the document chunks
    """
    
    def __init__(self, 
                 model_id: str = "gpt-4o", 
                 documents_dir: str = "./data/documents",
                 vector_store: VectorStore = None,
                 chunk_size: int = 512,
                 chunk_overlap: int = 128):
        """
        Initialize the Document Processor Agent.
        
        Args:
            model_id: ID of the OpenAI model to use
            documents_dir: Directory containing PDF documents
            vector_store: Vector store for document storage
            chunk_size: Maximum size of document chunks
            chunk_overlap: Overlap between chunks
        """
        # Create directories if they don't exist
        os.makedirs(documents_dir, exist_ok=True)
        
        # Set up agent config
        self.model_id = model_id
        self.documents_dir = documents_dir
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize document processor
        self.pdf_processor = PDFProcessor(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        
        # Create tools for the agent
        tools = self.get_tools()
        
        # Initialize the agent with specialized instructions
        self.agent = Agent(
            name="Document Processor",
            model=OpenAIChat(id=model_id),
            description="Specialized agent for document processing and knowledge base management",
            instructions=[
                "You are a document processing specialist responsible for building and managing a knowledge base.",
                
                "Step 1: Document Processing",
                "- When asked to process documents, scan the specified directory for PDF files",
                "- Extract and chunk text from each document while preserving metadata",
                "- Report any processing errors or issues encountered with specific files",
                
                "Step 2: Knowledge Base Management",
                "- Add processed document chunks to the vector store",
                "- Verify documents were successfully added to the knowledge base",
                "- Provide detailed information about the knowledge base contents when requested",
                
                "Important Guidelines:",
                "- Always handle files carefully and report any errors in detail",
                "- Provide clear summaries of processing results including document counts",
                "- If a processing request fails, explain the reason and suggest potential solutions",
                "- Never make up information about document processing results",
                
                "When reporting results:",
                "1. State whether the operation was successful",
                "2. Include the number of documents processed and chunks created",
                "3. List any files that failed processing and why",
                "4. Provide recommendations for fixing any issues"
            ],
            tools=tools,
            structured_outputs=ProcessingResult,
            markdown=True,
            debug_mode=True,
            show_tool_calls=True
        )
    
    def process_documents(self) -> ProcessingResult:
        """
        Process all PDF documents in the documents directory and add them to the vector store.
        
        Returns:
            ProcessingResult containing information about the processing operation
        """
        try:
            # Check if the documents directory exists and contains PDF files
            if not os.path.exists(self.documents_dir):
                return ProcessingResult(
                    success=False,
                    document_count=0,
                    chunk_count=0,
                    error_message=f"Documents directory not found: {self.documents_dir}"
                )
            
            # Get count of PDF files in the directory
            pdf_files = [f for f in os.listdir(self.documents_dir) if f.lower().endswith('.pdf')]
            if not pdf_files:
                return ProcessingResult(
                    success=False,
                    document_count=0,
                    chunk_count=0,
                    error_message=f"No PDF files found in {self.documents_dir}"
                )
            
            # Process PDF files using the PDF processor toolkit
            processing_result = self.pdf_processor.process_pdf_directory(
                self.documents_dir, 
                chunk_size=self.chunk_size, 
                chunk_overlap=self.chunk_overlap
            )
            
            if not processing_result["success"]:
                return ProcessingResult(
                    success=False,
                    document_count=0,
                    chunk_count=0,
                    error_message=processing_result.get("error", "Unknown processing error")
                )
            
            # Add document chunks to vector store using the vector store toolkit
            storage_result = self.vector_store.add_documents(processing_result["chunks"])
            
            # Return success result
            return ProcessingResult(
                success=True,
                document_count=len(pdf_files),
                chunk_count=len(processing_result["chunks"]),
                details=f"Successfully processed {len(pdf_files)} documents and created {len(processing_result['chunks'])} chunks"
            )
            
        except Exception as e:
            # Return error result
            return ProcessingResult(
                success=False,
                document_count=0,
                chunk_count=0,
                error_message=str(e)
            )
    
    def get_document_info(self) -> str:
        """
        Get information about the documents in the knowledge base.
        
        Returns:
            String with information about the documents
        """
        try:
            # Get count of items in the collection
            collection_count = self.vector_store.collection.count()
            
            # Get list of PDF files
            pdf_files = [f for f in os.listdir(self.documents_dir) if f.lower().endswith('.pdf')]
            
            # Format information
            info = f"Knowledge Base Information:\n"
            info += f"- Total documents: {len(pdf_files)}\n"
            info += f"- Total chunks in vector store: {collection_count}\n"
            info += f"- Document sources:\n"
            
            for pdf_file in pdf_files:
                file_path = os.path.join(self.documents_dir, pdf_file)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
                info += f"  - {pdf_file} ({file_size:.2f} MB)\n"
                
            return info
            
        except Exception as e:
            return f"Error getting document information: {str(e)}"

    def get_tools(self):
        """Get document processing tools for the Agno agent."""
        tools = [
            PDFProcessorToolkit(),
            VectorStoreToolkit(self.vector_store) if self.vector_store else None
        ]
        return [tool for tool in tools if tool]

# For backward compatibility
PDFDirectoryProcessorTool = PDFProcessorToolkit
PDFFileTool = PDFProcessorToolkit 