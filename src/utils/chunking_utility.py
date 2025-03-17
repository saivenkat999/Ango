import os
import logging
import re
from typing import List, Dict, Any, Optional
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
                
                # Extract document type and subject from file name
                doc_type, doc_subject = ChunkingUtility.extract_document_info(file_name, file_path)
                
                # Process each chunk with enhanced metadata
                for i, chunk in enumerate(document_chunks_agno):
                    # Extract meaningful chunk title/heading
                    chunk_title = ChunkingUtility.extract_chunk_title(chunk.content)
                    
                    # Extract key terms from the chunk
                    key_terms = ChunkingUtility.extract_key_terms(chunk.content)
                    
                    # Detect content type (code, table, text)
                    content_type = ChunkingUtility.detect_content_type(chunk.content)
                    
                    # Create a structured ID that contains meaningful information
                    chunk_id = f"{doc_type}_{doc_subject}_{content_type}_{i:03d}"
                    if chunk_title:
                        # Normalize the title for use in ID
                        normalized_title = re.sub(r'[^a-zA-Z0-9]', '_', chunk_title.lower())
                        normalized_title = re.sub(r'_+', '_', normalized_title)  # Replace multiple underscores with one
                        chunk_id = f"{doc_type}_{normalized_title}_{i:03d}"
                    
                    # Enhanced metadata with more structured information
                    enhanced_metadata = {
                        **chunk.meta_data,
                        "source": file_name,
                        "file_path": file_path,
                        "chunk_index": i,
                        "total_chunks": len(document_chunks_agno),
                        "file_type": agno_document.meta_data.get("file_type", "unknown"),
                        "document_type": doc_type,
                        "document_subject": doc_subject,
                        "content_type": content_type,
                        "chunk_title": chunk_title,
                        "key_terms": key_terms
                    }
                    
                    document_chunk = {
                        "id": chunk_id,
                        "text": chunk.content,
                        "metadata": enhanced_metadata
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
    
    @staticmethod
    def extract_document_info(file_name: str, file_path: str) -> tuple:
        """
        Extract document type and subject from file name.
        
        Args:
            file_name: Name of the document file
            file_path: Path to the document file
            
        Returns:
            Tuple of (doc_type, doc_subject)
        """
        # Default values
        doc_type = "documentation"
        doc_subject = "general"
        
        # Extract information from file name
        file_name_lower = file_name.lower()
        
        # Try to extract product name/type
        if "odbc" in file_name_lower:
            doc_type = "odbc"
        elif "jdbc" in file_name_lower:
            doc_type = "jdbc"
        elif "ado" in file_name_lower or "adonet" in file_name_lower:
            doc_type = "adonet"
        
        # Try to extract database/subject
        if "redshift" in file_name_lower:
            doc_subject = "redshift"
        elif "oracle" in file_name_lower:
            doc_subject = "oracle"
        elif "sqlserver" in file_name_lower or "sql-server" in file_name_lower:
            doc_subject = "sqlserver"
        elif "postgres" in file_name_lower:
            doc_subject = "postgres"
        elif "mysql" in file_name_lower:
            doc_subject = "mysql"
        
        return doc_type, doc_subject
    
    @staticmethod
    def extract_chunk_title(text: str) -> Optional[str]:
        """
        Extract a meaningful title from the chunk content.
        
        Args:
            text: The chunk text
            
        Returns:
            Extracted title or None if not found
        """
        # Try to find a heading-like structure
        heading_patterns = [
            r'^#+\s+(.+)$',  # Markdown heading
            r'^(.+)\n[=]+$',  # Markdown heading with = underline
            r'^(.+)\n[-]+$',  # Markdown heading with - underline
            r'^(\d+\.\d+\s+.+)$',  # Numbered section heading
            r'^(Chapter\s+\d+[\.:]\s+.+)$',  # Chapter heading
            r'^(Section\s+\d+[\.:]\s+.+)$',  # Section heading
        ]
        
        lines = text.split('\n')
        for i, line in enumerate(lines[:5]):  # Check first 5 lines
            # Check for potential headings
            line = line.strip()
            if not line:
                continue
                
            # Try all heading patterns
            for pattern in heading_patterns:
                match = re.match(pattern, line)
                if match:
                    return match.group(1).strip()
            
            # If first line is short and followed by blank line, it might be a title
            if i == 0 and len(line) < 80 and (len(lines) == 1 or not lines[1].strip()):
                return line
                
        # If no heading found, use first sentence if it's not too long
        first_sentence_match = re.match(r'^(.+?[.!?])\s', text)
        if first_sentence_match:
            first_sentence = first_sentence_match.group(1).strip()
            if len(first_sentence) < 100:
                return first_sentence
                
        return None
    
    @staticmethod
    def extract_key_terms(text: str) -> List[str]:
        """
        Extract key terms from the chunk content.
        
        Args:
            text: The chunk text
            
        Returns:
            List of key terms
        """
        # Simplified key term extraction - look for terms in quotes or capitalized phrases
        key_terms = []
        
        # Find terms in quotes
        quote_matches = re.findall(r'"([^"]+)"', text)
        for match in quote_matches:
            if 3 < len(match) < 50 and match not in key_terms:
                key_terms.append(match)
                
        # Find capitalized phrases (potential technical terms)
        cap_matches = re.findall(r'\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,3})\b', text)
        for match in cap_matches:
            if match not in key_terms and len(match) > 3:
                key_terms.append(match)
                
        # Limit to top 10 terms
        return key_terms[:10]
    
    @staticmethod
    def detect_content_type(text: str) -> str:
        """
        Detect the type of content in the chunk.
        
        Args:
            text: The chunk text
            
        Returns:
            Content type: 'code', 'table', or 'text'
        """
        # Check for code patterns
        code_patterns = [
            r'```[a-z]*\n.*?\n```',  # Markdown code blocks
            r'^\s*(?:def|class|function|var|let|const|import|public|private)\s+',  # Programming language keywords
            r'(?:\w+\(.*?\).*?{|\w+\s+=\s+function\s*\(.*?\))',  # Function definitions
            r'<\?(?:php|=).*?\?>',  # PHP code
            r'<script.*?>.*?</script>',  # JavaScript code
            r'<[a-z]+.*?>'  # HTML tags
        ]
        
        for pattern in code_patterns:
            if re.search(pattern, text, re.DOTALL | re.MULTILINE):
                return 'code'
                
        # Check for table patterns
        table_patterns = [
            r'\|\s*[-]+\s*\|',  # Markdown tables
            r'^\s*[+][-]+[+][-]+[+]$',  # ASCII tables
            r'<table.*?>.*?</table>',  # HTML tables
            r'\n\s*\|.*\|.*\n\s*\|.*\|'  # Simple pipe tables
        ]
        
        for pattern in table_patterns:
            if re.search(pattern, text, re.DOTALL | re.MULTILINE):
                return 'table'
                
        return 'text' 