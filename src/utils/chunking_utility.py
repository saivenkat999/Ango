import os
import logging
import re
import json
from typing import List, Dict, Any, Optional, Literal, Union
from pydantic import BaseModel, Field
from agno.document.base import Document as AgnoDocument
from agno.document.chunking.strategy import ChunkingStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ChunkingUtility")

# Enhanced metadata model with Pydantic for validation
class EnhancedMetadata(BaseModel):
    # Primary classification - required fields
    connection_type: Literal["jdbc", "odbc", "hdp", "api", "documentation"] = "documentation"
    database_type: str = "general"  # sqlserver, oracle, redshift, etc.
    
    # Content markers - helps with accurate retrieval
    keywords: List[str] = Field(default_factory=list)
    content_category: Literal["connection", "authentication", "querying", "configuration", "performance", "troubleshooting", "general"] = "general"
    
    # Specificity markers
    database_specific: bool = True  # Explicitly marks content as DB-specific
    connection_specific: bool = True  # Explicitly marks content as connection-type specific
    
    # Context information
    code_language: Optional[str] = None  # java, python, etc.
    relevant_versions: List[str] = Field(default_factory=list)
    
    # Document reference
    doc_id: str  # Unique identifier
    source_document: str  # Original document name/path
    chunk_id: Optional[str] = None  # ID within document if chunked
    chunk_index: int = 0
    total_chunks: int = 1
    
    # Original metadata fields (for backward compatibility)
    source: str
    file_path: str
    file_type: str = "unknown"
    
    # Extra fields
    extra: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"  # Allow extra fields that aren't in the model

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
                connection_type, database_type = ChunkingUtility.extract_document_info(file_name, file_path)
                
                # Process each chunk with enhanced metadata
                for i, chunk in enumerate(document_chunks_agno):
                    # Extract meaningful chunk title/heading
                    chunk_title = ChunkingUtility.extract_chunk_title(chunk.content)
                    
                    # Extract key terms from the chunk
                    key_terms = ChunkingUtility.extract_key_terms(chunk.content)
                    
                    # Detect content type (code, table, text)
                    content_type = ChunkingUtility.detect_content_type(chunk.content)
                    
                    # Detect content category
                    content_category = ChunkingUtility.detect_content_category(chunk.content)
                    
                    # Detect code language if applicable
                    code_language = ChunkingUtility.detect_code_language(chunk.content)
                    
                    # Extract relevant version information
                    relevant_versions = ChunkingUtility.extract_version_info(chunk.content)
                    
                    # Determine if content is database-specific or connection-specific
                    specificity = ChunkingUtility.detect_specificity(
                        chunk.content, 
                        connection_type, 
                        database_type
                    )
                    
                    # Create a structured ID that contains meaningful information
                    chunk_id = f"{connection_type}_{database_type}_{content_type}_{i:03d}"
                    if chunk_title:
                        # Normalize the title for use in ID
                        normalized_title = re.sub(r'[^a-zA-Z0-9]', '_', chunk_title.lower())
                        normalized_title = re.sub(r'_+', '_', normalized_title)  # Replace multiple underscores with one
                        chunk_id = f"{connection_type}_{normalized_title}_{i:03d}"
                    
                    # Create enhanced metadata with our new structure
                    enhanced_metadata = EnhancedMetadata(
                        connection_type=connection_type,
                        database_type=database_type,
                        keywords=key_terms,
                        content_category=content_category,
                        database_specific=specificity["database_specific"],
                        connection_specific=specificity["connection_specific"],
                        code_language=code_language,
                        relevant_versions=relevant_versions,
                        doc_id=f"{connection_type}_{database_type}_{os.path.splitext(file_name)[0]}",
                        source_document=file_name,
                        chunk_id=chunk_id,
                        chunk_index=i,
                        total_chunks=len(document_chunks_agno),
                        source=file_name,
                        file_path=file_path,
                        file_type=agno_document.meta_data.get("file_type", "unknown"),
                        # Include original metadata as extra
                        extra={
                            "document_type": connection_type,  # For backward compatibility
                            "document_subject": database_type,  # For backward compatibility
                            "content_type": content_type,
                            "chunk_title": chunk_title,
                        }
                    )
                    
                    # Convert to dict for storage in vector store
                    metadata_dict = enhanced_metadata.dict()
                    
                    document_chunk = {
                        "id": chunk_id,
                        "text": chunk.content,
                        "metadata": json.dumps(metadata_dict)  # Ensure metadata is properly serialized
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
        elif "hdp" in file_name_lower or "hybrid" in file_name_lower or "data pipeline" in file_name_lower:
            doc_type = "hdp"
        
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
    
    @staticmethod
    def detect_content_category(text: str) -> str:
        """
        Detect the content category of the chunk.
        
        Args:
            text: The chunk text
            
        Returns:
            Content category: connection, authentication, querying, configuration, performance, troubleshooting
        """
        text_lower = text.lower()
        
        # Define keyword patterns for each category
        patterns = {
            "connection": ["connect", "connection string", "connection url", "jdbc url", "odbc dsn", "datasource"],
            "authentication": ["auth", "login", "password", "credential", "token", "sso", "kerberos", "oauth"],
            "querying": ["query", "sql", "select", "insert", "update", "delete", "execute", "prepared statement"],
            "configuration": ["config", "property", "setting", "parameter", "option", "configure"],
            "performance": ["performance", "optimize", "tuning", "cache", "pool", "timeout", "concurrent"],
            "troubleshooting": ["error", "exception", "troubleshoot", "debug", "issue", "problem", "fail"]
        }
        
        # Count matches for each category
        category_scores = {category: 0 for category in patterns}
        for category, keywords in patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    category_scores[category] += 1
        
        # Find the category with the most matches
        max_score = 0
        best_category = "general"
        for category, score in category_scores.items():
            if score > max_score:
                max_score = score
                best_category = category
        
        return best_category
    
    @staticmethod
    def detect_code_language(text: str) -> Optional[str]:
        """
        Detect programming language in code blocks if present.
        
        Args:
            text: The chunk text
            
        Returns:
            Detected programming language or None
        """
        # Look for markdown code blocks with language specification
        code_blocks = re.findall(r'```(\w+)', text)
        if code_blocks:
            # Get the most common language
            languages = {}
            for lang in code_blocks:
                lang = lang.lower()
                # Skip if it's not a real language indicator
                if lang in ["", "none", "text", "markdown", "md"]:
                    continue
                languages[lang] = languages.get(lang, 0) + 1
            
            if languages:
                return max(languages.items(), key=lambda x: x[1])[0]
        
        # Look for language indicators in the text
        language_patterns = {
            "java": [r'public\s+class', r'import\s+java\.', r'System\.out\.', r'new\s+\w+\('],
            "python": [r'import\s+\w+', r'def\s+\w+\(', r'print\(', r'if\s+__name__\s+==\s+[\'"]__main__[\'"]'],
            "csharp": [r'using\s+System', r'namespace\s+\w+', r'public\s+class', r'static\s+void\s+Main'],
            "sql": [r'SELECT\s+.*\s+FROM', r'INSERT\s+INTO', r'UPDATE\s+.*\s+SET', r'CREATE\s+TABLE']
        }
        
        for lang, patterns in language_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return lang
        
        return None
    
    @staticmethod
    def extract_version_info(text: str) -> List[str]:
        """
        Extract version information from text.
        
        Args:
            text: The chunk text
            
        Returns:
            List of version strings
        """
        versions = []
        
        # Look for version patterns
        version_patterns = [
            # SQL Server versions
            (r'SQL\s+Server\s+(\d{4}(?:R2)?)', r'sqlserver_\1'),
            # Oracle versions
            (r'Oracle\s+(\d+(?:\.\d+)*[cCgG]?)', r'oracle_\1'),
            # JDBC versions
            (r'JDBC\s+(\d+(?:\.\d+)*)', r'jdbc_\1'),
            # ODBC versions
            (r'ODBC\s+(\d+(?:\.\d+)*)', r'odbc_\1'),
            # General version patterns
            (r'version\s+(\d+(?:\.\d+)*)', r'\1'),
            (r'v(\d+(?:\.\d+)*)', r'\1')
        ]
        
        for pattern, replacement in version_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                version = re.sub(pattern, replacement, match.group(0), flags=re.IGNORECASE).strip()
                if version not in versions:
                    versions.append(version)
        
        return versions
        
    @staticmethod
    def detect_specificity(text: str, connection_type: str, database_type: str) -> Dict[str, bool]:
        """
        Detect if content is specific to a connection type or database.
        
        Args:
            text: The chunk text
            connection_type: The detected connection type
            database_type: The detected database type
            
        Returns:
            Dictionary with specificity flags
        """
        text_lower = text.lower()
        
        # Patterns that indicate database-specific content
        db_specific_patterns = {
            "sqlserver": [
                r'sql\s+server', r'sqlserver', r'mssql', r't-sql', r'tsql',
                r'microsoft\s+sql', r'windows\s+authentication', r'sql\s+agent'
            ],
            "oracle": [
                r'oracle', r'pl\/sql', r'tns', r'tnsnames', r'oracledb',
                r'oracle\s+client', r'oci', r'oracle\s+net'
            ],
            "redshift": [
                r'redshift', r'amazon\s+redshift', r'spectrum', r'leader\s+node',
                r'workload\s+management', r'wlm', r'concurrency\s+scaling'
            ],
            "postgres": [
                r'postgres', r'postgresql', r'psql', r'pgadmin', r'pg_'
            ],
            "mysql": [
                r'mysql', r'innodb', r'myisam', r'my\.cnf', r'mysqladmin'
            ]
        }
        
        # Patterns that indicate connection-type specific content
        conn_specific_patterns = {
            "jdbc": [
                r'jdbc', r'java\.sql', r'connection\s+url', r'drivermanager',
                r'preparedstatement', r'resultset', r'datasource'
            ],
            "odbc": [
                r'odbc', r'sql_', r'sqlconnect', r'sqlexec', r'sqldrivers',
                r'dsn', r'driver\s+manager', r'odbcconf'
            ],
            "hdp": [
                r'hybrid\s+data\s+pipeline', r'hdp', r'progress\s+datadirect', r'data\s+pipeline', 
                r'on-premises\s+connector', r'odata', r'data\s+gateway', r'single\s+sign-on',
                r'identity\s+management', r'universal\s+client'
            ]
        }
        
        # Check if content has database-specific indicators
        db_specific = False
        if database_type in db_specific_patterns:
            patterns = db_specific_patterns[database_type]
            matches = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            
            # If matching more than 2 patterns or over 25% of the patterns, consider it database-specific
            db_specific = matches >= 2 or (matches / len(patterns) >= 0.25)
            
        # Check for general database-specific content
        if not db_specific:
            # If mentions multiple databases, it's likely not specific to one
            db_mentions = sum(1 for db in db_specific_patterns if any(re.search(pattern, text_lower) for pattern in db_specific_patterns[db]))
            db_specific = db_mentions <= 1 and db_mentions > 0
            
        # Check if content has connection-type-specific indicators
        conn_specific = False
        if connection_type in conn_specific_patterns:
            patterns = conn_specific_patterns[connection_type]
            matches = sum(1 for pattern in patterns if re.search(pattern, text_lower))
            
            # If matching more than 2 patterns or over 25% of the patterns, consider it connection-specific
            conn_specific = matches >= 2 or (matches / len(patterns) >= 0.25)
            
        # Check for general connection-type-specific content
        if not conn_specific:
            # If mentions multiple connection types, it's likely not specific to one
            conn_mentions = sum(1 for conn in conn_specific_patterns if any(re.search(pattern, text_lower) for pattern in conn_specific_patterns[conn]))
            conn_specific = conn_mentions <= 1 and conn_mentions > 0
        
        return {
            "database_specific": db_specific,
            "connection_specific": conn_specific
        } 