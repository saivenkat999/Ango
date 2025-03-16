import os
import glob
import logging
from typing import List, Dict, Any, Optional
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import Toolkit
from pydantic import BaseModel, Field
from src.utils.vector_store import VectorStore
from agno.document.chunking.semantic import SemanticChunking
from agno.document.chunking.recursive import RecursiveChunking
from agno.document.chunking.document import DocumentChunking
from src.utils.pdf_parser import PDFProcessor
from src.utils.chunking_utility import ChunkingUtility
from src.utils.config import Config
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
    chunks: Optional[List[Dict[str, Any]]] = Field(None, description="List of processed document chunks")
    failed_files: Optional[List[Dict[str, Any]]] = Field(None, description="List of failed files with errors")

class DocumentProcessorToolkit(Toolkit):
    """Toolkit for processing PDF documents."""

    def __init__(self):
        super().__init__(name="document_processor")
        self.register(self.process_documents)

        # Get chunking strategy from environment variable
        chunking_strategy = os.getenv("CHUNKING_STRATEGY", "semantic").lower()
        chunk_size = int(os.getenv("CHUNK_SIZE", "1024"))

        # Initialize the appropriate chunking strategy
        if chunking_strategy == "semantic":
            self.chunking_strategy = SemanticChunking(chunk_size=chunk_size)
        elif chunking_strategy == "document":
            self.chunking_strategy = DocumentChunking(chunk_size=chunk_size)
        elif chunking_strategy == "recursive":
            self.chunking_strategy = RecursiveChunking(chunk_size=chunk_size)
        else:
            logger.warning(f"Unknown chunking strategy '{chunking_strategy}', using 'semantic' instead")
            self.chunking_strategy = SemanticChunking(chunk_size=chunk_size)

        logger.info(f"Using chunking strategy: {self.chunking_strategy.__class__.__name__}")

        # Initialize PDF processor
        self.pdf_processor = PDFProcessor()

    def process_documents(self, path: str) -> Dict[str, Any]:
        """
        Process PDF document(s) from a path and return the chunks.

        Args:
            path: Path to PDF document file or directory

        Returns:
            Dict containing status and results
        """
        logger.info(f"Processing PDF document(s) from: {path}")

        # Check if path exists
        if not os.path.exists(path):
            logger.error(f"Path does not exist: {path}")
            return {
                "success": False,
                "error": f"Path {path} does not exist",
                "document_count": 0,
                "chunk_count": 0
            }

        try:
            # Process the PDF document(s)
            result = self.pdf_processor.process_documents(path)

            if not result["success"]:
                return {
                    "success": False,
                    "error": result["error"],
                    "failed_files": result.get("failed_files"),
                    "document_count": 0,
                    "chunk_count": 0
                }

            # Apply chunking strategy to documents
            chunking_result = ChunkingUtility.chunk_documents(
                result["documents"],
                self.chunking_strategy
            )

            # Include any failed files from processing step
            if result.get("failed_files"):
                chunking_result["failed_files"] = result.get("failed_files", [])

            return chunking_result

        except Exception as e:
            logger.error(f"Error processing path {path}: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Error processing path {path}: {str(e)}",
                "document_count": 0,
                "chunk_count": 0
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
            if self.vector_store.collection_name in self.vector_store.client.table_names():
                table = self.vector_store.client.open_table(self.vector_store.collection_name)
                collection_count = len(table.to_pandas())
                return {
                    "success": True,
                    "collection_name": self.vector_store.collection_name,
                    "document_count": collection_count,
                    "details": f"Collection '{self.vector_store.collection_name}' contains {collection_count} document chunks"
                }
            else:
                return {
                    "success": False,
                    "error": f"Collection '{self.vector_store.collection_name}' does not exist"
                }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": f"Error getting collection info: {str(e)}"
            }

class DocumentProcessorAgent:
    """Agent for processing PDF documents."""

    def __init__(self, model_id: str = "gpt-4o",
                 documents_dir: str = "./data/documents",
                 vector_store: VectorStore = None):
        self.model_id = model_id
        self.documents_dir = documents_dir
        self.vector_store = vector_store
        self.vector_store_toolkit = VectorStoreToolkit(vector_store)
        self.document_processor = DocumentProcessorToolkit()

        # Initialize the agent with detailed instructions
        self.agent = Agent(
            name="Document Processor",
            model=OpenAIChat(id=self.model_id, api_key=Config.OPENAI_API_KEY),
            description="Specialized agent for processing and analyzing PDF documents",
            instructions=[
                "You are an expert document processor responsible for analyzing and processing PDF documents.",
                "Your tasks include:",
                "1. Processing PDF documents using LlamaCloud Parsing",
                "2. Applying appropriate chunking strategies",
                "3. Managing the vector store for document storage",
                "4. Handling various document structures",
                "5. Ensuring proper metadata extraction and storage",

                "Important Guidelines:",
                "- Process documents efficiently and accurately",
                "- Handle errors gracefully and provide clear feedback",
                "- Maintain document integrity during processing",
                "- Apply appropriate chunking based on content type",
                "- Ensure proper vector store management"
            ],
            tools=[self.document_processor, self.vector_store_toolkit],
            markdown=True,
            debug_mode=True,
            show_tool_calls=True
        )

    def process_and_store_document(self, document_path: str) -> ProcessingResult:
        """Process a PDF document and store it in the vector store."""
        try:
            # Process the document
            result = self.document_processor.process_documents(document_path)

            if not result["success"]:
                return ProcessingResult(
                    success=False,
                    document_count=0,
                    chunk_count=0,
                    error_message=result["error"],
                    failed_files=result.get("failed_files")
                )

            # Store chunks in vector store
            chunks = result["chunks"]
            store_result = self.vector_store_toolkit.add_documents(chunks)

            if not store_result["success"]:
                return ProcessingResult(
                    success=False,
                    document_count=result["document_count"],
                    chunk_count=0,
                    error_message=f"Failed to store chunks: {store_result.get('error', 'Unknown error')}",
                    failed_files=result.get("failed_files")
                )

            return ProcessingResult(
                success=True,
                document_count=result["document_count"],
                chunk_count=result["chunk_count"],
                details=f"Successfully processed and stored {result['chunk_count']} chunks from {result['document_count']} documents",
                chunks=chunks,
                failed_files=result.get("failed_files")
            )

        except Exception as e:
            logger.error(f"Error processing document {document_path}: {str(e)}", exc_info=True)
            return ProcessingResult(
                success=False,
                document_count=0,
                chunk_count=0,
                error_message=f"Error processing document: {str(e)}"
            )

    def get_document_info(self) -> str:
        """Get information about the documents in the vector store."""
        try:
            collection_info = self.vector_store_toolkit.get_collection_info()

            if not collection_info["success"]:
                return f"Error retrieving collection info: {collection_info.get('error', 'Unknown error')}"

            # Get available PDF documents
            available_documents = glob.glob(os.path.join(self.documents_dir, "*.pdf"))

            # Format the information
            info = [
                "# Document Knowledge Base Information",
                "",
                f"## Vector Store: {collection_info['collection_name']}",
                f"- Total document chunks: {collection_info['document_count']}",
                f"- Chunking strategy: {self.document_processor.chunking_strategy.__class__.__name__}",
                "",
                "## Available PDF Documents",
            ]

            if available_documents:
                for doc in available_documents:
                    info.append(f"- {os.path.basename(doc)}")
            else:
                info.append("- No PDF documents available in the documents directory")

            return "\n".join(info)

        except Exception as e:
            logger.error(f"Error getting document info: {str(e)}", exc_info=True)
            return f"Error getting document information: {str(e)}"