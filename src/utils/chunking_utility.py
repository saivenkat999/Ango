import os
import logging
from typing import List, Dict, Any
from agno.document.base import Document as AgnoDocument
from agno.document.chunking.strategy import ChunkingStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ChunkingUtility")

class ChunkingUtility:
    """
    Utility for applying chunking strategies to documents and formatting the results.
    
    This class provides methods to:
    1. Apply chunking strategies to Agno Document objects
    2. Format chunked documents into a standardized dictionary format
    3. Handle errors and provide detailed operation results
    """
    
    @staticmethod
    def chunk_documents(agno_documents: List[AgnoDocument], chunking_strategy: ChunkingStrategy) -> Dict[str, Any]:
        """
        Apply chunking strategy to a list of Agno documents and format results.
        
        Args:
            agno_documents: List of Agno Document objects to process
            chunking_strategy: Chunking strategy to apply
            
        Returns:
            Dict containing status and results with formatted chunks
        """
        if not agno_documents:
            return {
                "success": False,
                "error": "No documents to process",
                "document_count": 0,
                "chunk_count": 0
            }
        
        all_chunks = []
        processed_files = []
        failed_files = []
        
        for agno_document in agno_documents:
            try:
                # Apply chunking strategy
                document_chunks_agno = chunking_strategy.chunk(agno_document)
                
                # Convert chunks to expected format
                file_path = agno_document.meta_data["file_path"]
                file_name = os.path.basename(file_path)
                
                for i, chunk in enumerate(document_chunks_agno):
                    document_chunk = {
                        "id": chunk.id if chunk.id else f"{file_name}_{i}",
                        "text": chunk.content,
                        "metadata": {
                            **chunk.meta_data,
                            "source": file_name,
                            "file_path": file_path,
                            "chunk_index": i,
                            "total_chunks": len(document_chunks_agno),
                            "file_type": agno_document.meta_data.get("file_type", "unknown")
                        }
                    }
                    all_chunks.append(document_chunk)
                
                processed_files.append(file_name)
                logger.debug(f"Successfully chunked {file_name} into {len(document_chunks_agno)} chunks")
                
            except Exception as e:
                logger.error(f"Error chunking document {agno_document.name}: {str(e)}", exc_info=True)
                failed_files.append({
                    "file": agno_document.name,
                    "error": str(e)
                })
        
        # Return results
        if not processed_files:
            return {
                "success": False,
                "error": "Failed to process any files",
                "failed_files": failed_files,
                "document_count": 0,
                "chunk_count": 0
            }
        
        return {
            "success": True,
            "chunks": all_chunks,
            "processed_files": processed_files,
            "failed_files": failed_files if failed_files else None,
            "document_count": len(processed_files),
            "chunk_count": len(all_chunks),
            "details": f"Successfully processed {len(processed_files)} files with {len(all_chunks)} total chunks"
        } 