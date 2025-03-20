import logging
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import Toolkit
from pydantic import BaseModel, Field
from ..utils.config import Config
import json
import re
from functools import lru_cache

from ..utils.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RetrieverAgent")

class RetrievalQuery(BaseModel):
    """Structured query for retrieving information from the knowledge base."""
    query: str = Field(description="The query text to search for")
    n_results: int = Optional[Field(default=5, description="Number of results to retrieve")]
    metadata_filter: Optional[Any] = Field(None, description="Optional metadata filter")
    search_type: str = Optional[Field(default="hybrid", description="Search type to use (hybrid, vector, or text)")]
    expected_result_type: str = Optional[Field(default="factual", description="Expected result type (factual, conceptual, procedural, opinion)")]
    query_context: Optional[str] = Field(None, description="Additional context about the query to aid in retrieval")

    class Config:
        extra = 'forbid'  # This ensures no additional fields are allowed

class RetrievalResult(BaseModel):
    """Structured result from knowledge base retrieval operations."""
    success: bool = Field(description="Whether the retrieval was successful")
    query: str = Field(description="The original query")
    results: List[Dict[str, Any]] = Field(description="The retrieved results")
    context: Optional[str] = Field(None, description="Formatted context from the results")
    error_message: Optional[str] = Field(None, description="Error message if retrieval failed")

class KeywordExtractor:
    """
    Utility for extracting document-specific keywords from queries.
    
    This class maintains mappings of keywords to document types and
    helps identify which document is being referred to in a query.
    """
    
    def __init__(self):
        """Initialize keyword mappings for different document types."""
        # Technology-specific keywords (connection types)
        self.connection_type_keywords = {
            "odbc": {"odbc", "open database connectivity", "dsn", "data source name", "driver manager"},
            "jdbc": {"jdbc", "java database", "connection pool", "java", "j2ee", "connection url"},
            "hdp": {"hdp", "hybrid data pipeline", "data pipeline", "progress datadirect", "odata", "data gateway", "universal connectivity", "on-premises connector"},
        }
        
        # Database-specific keywords
        self.database_type_keywords = {
            "redshift": {"redshift", "amazon redshift", "aws", "columnar", "mpp", "massively parallel"},
            "oracle": {"oracle", "pl/sql", "tns", "tnsnames", "tablespace", "oracle database"},
            "sqlserver": {"sql server", "microsoft sql", "t-sql", "tsql", "mssql", "windows authentication"},
            "postgres": {"postgres", "postgresql", "psql", "pg"},
            "mysql": {"mysql", "my.cnf", "innodb", "myisam"},
        }
        
        # Content category keywords
        self.content_category_keywords = {
            "connection": {"connect", "connection", "setup", "configure", "establish", "datasource", "connectionstring"},
            "authentication": {"auth", "authentication", "login", "credential", "password", "username", "sso", "identity"},
            "querying": {"query", "sql", "select", "execute", "statement", "prepared statement", "result", "fetch"},
            "configuration": {"config", "property", "setting", "parameter", "option", "driver", "setup", "registry"},
            "performance": {"performance", "tuning", "optimize", "cache", "connection pool", "pooling", "timeout"},
            "troubleshooting": {"error", "troubleshoot", "debug", "issue", "problem", "failure", "diagnostic", "exception"},
        }
        
        # Backward compatibility mappings
        self.tech_keywords = self.connection_type_keywords
        self.db_keywords = self.database_type_keywords
        self.feature_keywords = {
            "encryption": {"encrypt", "encryption", "ssl", "tls", "secure", "security", "certificate"},
            "authentication": self.content_category_keywords["authentication"],
            "performance": self.content_category_keywords["performance"],
            "troubleshooting": self.content_category_keywords["troubleshooting"],
        }

    def extract_keywords(self, query: str) -> Dict[str, Set[str]]:
        """
        Extract keywords from the query that match known document types.
        
        Args:
            query: The query string to analyze
            
        Returns:
            Dictionary with categories as keys and sets of matched keywords as values
        """
        query = query.lower()
        results = {
            "connection_type": set(),  # New field name
            "database_type": set(),    # New field name
            "content_category": set(),  # New field
            # Keep old field names for backward compatibility
            "technology": set(),
            "database": set(),
            "feature": set()
        }
        
        # Check if this is a comparative query
        is_comparative = any(indicator in query for indicator in [
            " vs ", " versus ", " compare ", " difference between ", " or ", " and "
        ])
        
        # For comparative queries, be more selective with keyword matching
        if is_comparative:
            # For comparative queries, only extract connection_type and database_type
            # without adding additional content categories or features
            
            # Check for database types directly - higher precision for comparative queries
            database_types = get_common_patterns("database_types")
            for db_type in database_types:
                # Check for exact database names in comparative queries
                if f" {db_type} " in f" {query} " or f"{db_type}," in query or query.endswith(f" {db_type}"):
                    results["database_type"].add(db_type)
                    results["database"].add(db_type)  # For backward compatibility
            
            # Check for connection types directly - higher precision for comparative queries
            connection_types = get_common_patterns("connection_types")
            for conn_type in connection_types:
                # Check for exact connection types in comparative queries
                if f" {conn_type} " in f" {query} " or f"{conn_type}," in query or query.endswith(f" {conn_type}"):
                    results["connection_type"].add(conn_type)
                    results["technology"].add(conn_type)  # For backward compatibility
            
            return results
        
        # Non-comparative queries use the original algorithm
        
        # Check for connection type keywords
        for tech, keywords in self.connection_type_keywords.items():
            if any(kw in query for kw in keywords):
                results["connection_type"].add(tech)
                results["technology"].add(tech)  # For backward compatibility
                
        # Check for database keywords
        for db, keywords in self.database_type_keywords.items():
            if any(kw in query for kw in keywords):
                results["database_type"].add(db)
                results["database"].add(db)  # For backward compatibility
                
        # Check for content category keywords
        for category, keywords in self.content_category_keywords.items():
            if any(kw in query for kw in keywords):
                results["content_category"].add(category)
                
        # For backward compatibility: check for feature keywords
        for feature, keywords in self.feature_keywords.items():
            if any(kw in query for kw in keywords):
                results["feature"].add(feature)
                
        # Attempt direct detection for common patterns if other methods failed
        if not results["connection_type"]:
            # Try direct pattern matching for connection types
            connection_types = get_common_patterns("connection_types")
            for conn_type in connection_types:
                if conn_type in query or conn_type.upper() in query:
                    results["connection_type"].add(conn_type)
                    results["technology"].add(conn_type)  # For backward compatibility
        
        if not results["database_type"]:
            # Try direct pattern matching for database types
            database_types = get_common_patterns("database_types")
            for db_type in database_types:
                if db_type in query or db_type.capitalize() in query:
                    results["database_type"].add(db_type)
                    results["database"].add(db_type)  # For backward compatibility
                    
        return results
    
    def enhance_metadata_filter(self, query: str, metadata_filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance metadata filter with extracted keywords from the query.
        
        Args:
            query: The query string to analyze
            metadata_filter: Existing metadata filter to enhance
            
        Returns:
            Enhanced metadata filter
        """
        # Start with a copy of the existing filter
        enhanced_filter = metadata_filter.copy() if metadata_filter else {}
        
        # Extract keywords from the query
        extracted = self.extract_keywords(query)
        
        # For comparative queries, don't apply strict database filters
        query_lower = query.lower()
        if any(indicator in query_lower for indicator in [
            " vs ", " versus ", " compare ", " difference between ", " or ", " and "
        ]):
            # Detected a likely comparative query
            logger.info("Detected comparative query, using relaxed metadata filtering")
            
            # Only add connection_type for comparative queries if found
            if extracted["connection_type"] and "connection_type" not in enhanced_filter:
                connection_type = next(iter(extracted["connection_type"]))
                enhanced_filter["connection_type"] = connection_type
                
            # Do not restrict by database_type for comparative queries
            if "database_type" in enhanced_filter:
                logger.info(f"Removing database_type filter for comparative query")
                del enhanced_filter["database_type"]
                
            return enhanced_filter
        
        # For non-comparative queries, apply normal filtering
        
        # Update connection_type if found in query and not already set
        if extracted["connection_type"] and "connection_type" not in enhanced_filter:
            # Use the first one found (most likely match)
            connection_type = next(iter(extracted["connection_type"]))
            enhanced_filter["connection_type"] = connection_type
            logger.debug(f"Added connection_type filter: {connection_type}")
            
            # For backward compatibility
            if "technology" not in enhanced_filter:
                enhanced_filter["technology"] = connection_type
        
        # Update database_type if found in query and not already set
        if extracted["database_type"] and "database_type" not in enhanced_filter:
            # Use the first one found (most likely match)
            database_type = next(iter(extracted["database_type"]))
            enhanced_filter["database_type"] = database_type
            logger.debug(f"Added database_type filter: {database_type}")
            
            # For backward compatibility
            if "database" not in enhanced_filter:
                enhanced_filter["database"] = database_type
        
        # Update content_category if found and not already set
        if extracted["content_category"] and "content_category" not in enhanced_filter:
            # Use the first one found (most likely match)
            content_category = next(iter(extracted["content_category"]))
            enhanced_filter["content_category"] = content_category
            logger.debug(f"Added content_category filter: {content_category}")
            
        # For backward compatibility: add feature if found and not set
        if extracted["feature"] and "feature" not in enhanced_filter:
            # Use the first one found (most likely match)
            feature = next(iter(extracted["feature"]))
            enhanced_filter["feature"] = feature
            logger.debug(f"Added feature filter: {feature}")
            
        return enhanced_filter

    def enhance_query(self, query_text: str, metadata_filter: Dict[str, Any]) -> str:
        """
        Enhance the query text with additional context based on metadata filters.
        
        Args:
            query_text: The original query text
            metadata_filter: Metadata filters to derive context from
            
        Returns:
            Enhanced query text with additional context
        """
        # Check if this is a comparative query - if so, don't enhance
        query_lower = query_text.lower()
        if any(indicator in query_lower for indicator in [
            " vs ", " versus ", " compare ", " difference between ", " or ", " and "
        ]):
            logger.info("Detected comparative query, skipping query enhancement")
            return query_text
            
        enhanced_query = query_text
        
        # Extract key context from metadata filters
        connection_type = metadata_filter.get("connection_type")
        database_type = metadata_filter.get("database_type")
        content_category = metadata_filter.get("content_category")
        
        # Don't modify the query if it already contains these terms
        
        # Add connection type context if not already in query
        if connection_type and connection_type not in query_lower:
            # For HDP, use the full name for better context
            if connection_type == "hdp" and "hybrid data pipeline" not in query_lower:
                enhanced_query += f" for Hybrid Data Pipeline"
            elif connection_type not in query_lower:
                enhanced_query += f" for {connection_type.upper()}"
        
        # Add database type context if not already in query
        if database_type and database_type not in query_lower:
            enhanced_query += f" with {database_type}"
        
        # Add content category context for certain categories
        if content_category and content_category not in query_lower:
            if content_category == "authentication":
                enhanced_query += " for authentication purposes"
            elif content_category == "connection":
                enhanced_query += " when establishing connections"
            elif content_category == "configuration":
                enhanced_query += " configuration"
        
        logger.debug(f"Enhanced query from '{query_text}' to '{enhanced_query}'")
        return enhanced_query

class InfoRetrievalToolkit(Toolkit):
    """
    Toolkit for retrieving information from the knowledge base using advanced search techniques.

    This toolkit provides methods to search through document collections using:
    - Vector search (semantic similarity)
    - Keyword search (text matching)
    - Hybrid search (combination of vector and keyword)

    It also applies reranking to improve result quality and relevance.

    Use this toolkit when you need to find specific information from stored documents
    based on a user query.
    """

    def __init__(self, vector_store: VectorStore):
        super().__init__(name="information_retrieval")
        self.vector_store = vector_store

        # Initialize the keyword extractor
        self.keyword_extractor = KeywordExtractor()

        # A simple cache for recently retrieved information
        self.query_cache = {}
        self.max_cache_size = 20  # Maximum number of cached queries

        # We no longer need to initialize our own reranker,
        # since we'll use the one from the vector_store
        # which is already configured according to Config.RERANKER_TYPE
        logger.info(f"Using reranker from VectorStore ({Config.RERANKER_TYPE})")

        # Register the retrieve_information function
        self.register(self.retrieve_information)
        
    def decompose_complex_query(self, query_text: str) -> List[str]:
        """
        Decompose a complex query into simpler subqueries for better results.
        
        Args:
            query_text: The original query text
            
        Returns:
            List of subqueries, or just the original query if decomposition not needed
        """
        # Simple rule-based decomposition - look for indicators of complex queries
        query_lower = query_text.lower()
        
        # For queries specifically about authentication methods between database types,
        # return the original query to ensure we search broadly across all databases
        if "authentication" in query_lower and any(db_type in query_lower for db_type in get_common_patterns("database_types")):
            if any(comp in query_lower for comp in [" vs ", " versus ", " compare ", " difference ", " and ", " or "]):
                logger.info("Detected database authentication comparison query, keeping as single query")
                return [query_text]
        
        # Check if the query contains multiple distinct questions
        multiple_question_indicators = [
            " and also ", " also ", " additionally ", 
            " what about ", " how about ", " moreover ",
            "? ", "; "
        ]
        
        # Check if query contains comparative elements
        comparative_indicators = [
            " vs ", " versus ", " compare ", " difference between ",
            " pros and cons ", " advantages and disadvantages "
        ]
        
        # Check for decomposition needs
        needs_decomposition = False
        
        # Multiple questions in one
        if any(indicator in query_lower for indicator in multiple_question_indicators):
            needs_decomposition = True
            
        # Comparative questions
        if any(indicator in query_lower for indicator in comparative_indicators):
            needs_decomposition = True
            
        # If we don't need decomposition, return the original query
        if not needs_decomposition:
            return [query_text]
            
        # Let's decompose based on the query type
        
        # First try to split by question marks
        if "?" in query_text:
            subqueries = [q.strip() + "?" for q in query_text.split("?") if q.strip()]
            if len(subqueries) > 1:
                logger.info(f"Decomposed query into {len(subqueries)} subqueries based on question marks")
                return subqueries
        
        # For database comparisons, keep as a single query rather than decomposing
        # This ensures we get broad results that can be compared properly
        database_types = get_common_patterns("database_types") 
        mentioned_dbs = [db for db in database_types if db in query_lower]
        if len(mentioned_dbs) > 1 and any(comp in query_lower for comp in comparative_indicators):
            logger.info("Query compares multiple databases, keeping as a single query")
            return [query_text]
                
        # For comparative queries, break into two separate queries about each entity
        for indicator in comparative_indicators:
            if indicator in query_lower:
                parts = query_lower.split(indicator)
                if len(parts) == 2:
                    # Extract topics being compared
                    topic1 = parts[0].strip()
                    topic2 = parts[1].strip()
                    
                    # For "difference between A and B" pattern
                    if indicator == " difference between " and " and " in topic2:
                        topic2_parts = topic2.split(" and ", 1)
                        if len(topic2_parts) == 2:
                            topic1 = topic2_parts[0].strip()
                            topic2 = topic2_parts[1].strip()
                    
                    # Create focused queries for each topic
                    subquery1 = f"Tell me about {topic1}"
                    subquery2 = f"Tell me about {topic2}"
                    
                    logger.info(f"Decomposed comparative query into two topic-specific subqueries")
                    return [subquery1, subquery2]
        
        # If we can't decompose in a smart way, just return the original
        return [query_text]
        
    def retrieve_information(self, retrieval_query: RetrievalQuery) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve relevant information from the vector store based on the query.

        This method:
        1. Searches the vector database using the specified search type
        2. Applies reranking to improve result quality
        3. Formats results based on the expected result type

        Args:
            retrieval_query: A structured query object containing:
                - query: The text to search for
                - n_results: Number of results to retrieve (default: 5)
                - metadata_filter: Optional filters to apply to search results
                - search_type: Search strategy to use (hybrid, vector, text)
                - expected_result_type: Type of result expected (factual, conceptual, etc.)
                - query_context: Additional context about the query

        Returns:
            A tuple containing:
                - formatted_context: A string containing formatted retrieved information
                - document_results: The raw document results list

        Example:
            To find information about machine learning:
            retrieve_information(RetrievalQuery(
                query="What is machine learning?",
                n_results=3,
                search_type="hybrid",
                expected_result_type="conceptual"
            ))
        """
        try:
            # Extract query text and other parameters
            query_text = retrieval_query.query
            metadata_filter = retrieval_query.metadata_filter or {}
            
            # Check if this is a complex query that should be decomposed
            subqueries = self.decompose_complex_query(query_text)
            
            # If we have multiple subqueries, handle each separately and combine results
            if len(subqueries) > 1:
                logger.info(f"Processing complex query as {len(subqueries)} subqueries")
                all_results = []
                all_formatted = []
                
                # Process each subquery with a smaller result count
                n_per_query = max(2, retrieval_query.n_results // len(subqueries))
                
                for i, subquery in enumerate(subqueries):
                    logger.info(f"Processing subquery {i+1}/{len(subqueries)}: '{subquery}'")
                    # Create a new retrieval query for this subquery
                    subquery_retrieval = RetrievalQuery(
                        query=subquery,
                        n_results=n_per_query,
                        metadata_filter=metadata_filter,
                        search_type=retrieval_query.search_type,
                        expected_result_type=retrieval_query.expected_result_type,
                        query_context=retrieval_query.query_context
                    )
                    
                    # Get results for this subquery
                    formatted_context, document_results = self._process_single_query(subquery_retrieval)
                    
                    if document_results:
                        # Add subquery indicator to the formatted context
                        subquery_header = f"**Subquery {i+1}: {subquery}**\n\n"
                        all_formatted.append(subquery_header + formatted_context)
                        all_results.extend(document_results)
                
                # Combine all results
                if not all_results:
                    return "No information related to your query was found in the knowledge base.", []
                
                # Return the combined results
                combined_context = "\n\n".join(all_formatted)
                return combined_context, all_results[:retrieval_query.n_results]
            
            # For single queries, process normally
            return self._process_single_query(retrieval_query)
            
        except Exception as e:
            logger.error(f"Error retrieving information: {str(e)}", exc_info=True)
            error_message = f"Error retrieving information: {str(e)}"
            return error_message, []
            
    def _process_single_query(self, retrieval_query: RetrievalQuery) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Process a single query for information retrieval.
        
        This is the core implementation that handles a simple non-decomposed query.
        
        Args:
            retrieval_query: The query to process
            
        Returns:
            Tuple of (formatted_context, document_results)
        """
        try:
            # Extract query text and other parameters
            query_text = retrieval_query.query
            metadata_filter = retrieval_query.metadata_filter or {}
            
            # Check cache first for identical query with same metadata filters
            cache_key = f"{query_text}_{json.dumps(metadata_filter, sort_keys=True)}"
            if cache_key in self.query_cache:
                logger.info(f"Cache hit for query: '{query_text}'")
                cached_result = self.query_cache[cache_key]
                return cached_result[0], cached_result[1]
            
            # Check if this is a comparative query - if so, request more results
            query_lower = query_text.lower()
            is_comparative = any(indicator in query_lower for indicator in [
                " vs ", " versus ", " compare ", " difference between ", " or ", " and "
            ])
            
            # Use more results for comparative queries to get better coverage
            n_results = retrieval_query.n_results
            if is_comparative:
                n_results = max(n_results * 2, 10)  # Double results, minimum 10
                logger.info(f"Detected comparative query, increasing result count to {n_results}")
            
            # Enhance metadata filters with keyword extraction
            enhanced_filter = self.keyword_extractor.enhance_metadata_filter(query_text, metadata_filter)
            
            # Log the metadata filter updates
            if enhanced_filter != metadata_filter:
                logger.info(f"Enhanced metadata filters from: {metadata_filter} to: {enhanced_filter}")
                metadata_filter = enhanced_filter
            
            # Enhance the query with additional context from metadata
            enhanced_query = self.keyword_extractor.enhance_query(query_text, metadata_filter)
            
            # Use the enhanced query if it's significantly different
            if len(enhanced_query) > len(query_text) + 5:  # Only use if meaningful additions were made
                logger.info(f"Using enhanced query: '{enhanced_query}'")
                query_for_search = enhanced_query
            else:
                query_for_search = query_text
            
            # Log the retrieval request
            logger.info(f"Retrieving information for query: '{query_for_search}', n_results={n_results}, filters={metadata_filter}")

            # The VectorStore returns a list of document dictionaries
            document_results = self.vector_store.query(
                query_text=query_for_search,
                n_results=n_results,
                metadata_filter=metadata_filter,
                search_type=retrieval_query.search_type,
            )

            # Check if we have any results
            if not document_results or len(document_results) == 0:
                logger.warning(f"No results found for query: {query_for_search}")
                return "No information related to your query was found in the knowledge base.", []

            # Defensive validation of document_results to ensure proper format
            validated_results = []
            for i, doc_dict in enumerate(document_results):
                # Skip if not a dictionary
                if not isinstance(doc_dict, dict):
                    logger.warning(f"Skipping non-dictionary result at index {i}: {type(doc_dict)}")
                    continue
                
                # Ensure the dictionary has the required keys
                if 'text' not in doc_dict or not doc_dict['text']:
                    logger.warning(f"Skipping result missing 'text' at index {i}")
                    continue
                
                # Ensure all required fields have valid values
                valid_doc = {
                    'id': doc_dict.get('id', f"doc_{i}"),
                    'text': doc_dict['text'],
                    'metadata': doc_dict.get('metadata', {}),
                    'score': doc_dict.get('score', 0.0)
                }
                validated_results.append(valid_doc)

            # If validation removed all results
            if not validated_results:
                logger.warning(f"No valid results remained after validation for query: {query_for_search}")
                return "No valid information related to your query was found in the knowledge base.", []
                
            document_results = validated_results
            logger.debug(f"Validated {len(document_results)} document results")
            
            # Rerank documents using vector_store if needed
            if Config.USE_RERANKING:
                # Use the VectorStore's dedicated reranking method
                reranked_results = self.vector_store.rerank_documents(
                    query_text=query_for_search,
                    documents=document_results,
                    n_results=retrieval_query.n_results if not is_comparative else n_results
                )
                
                # If reranking was successful, use the reranked results
                if reranked_results:
                    document_results = reranked_results
                    logger.info(f"Documents reranked using VectorStore's {Config.RERANKER_TYPE} reranker")
            
            # Format the context
            formatted_context = self._format_context(document_results, retrieval_query.expected_result_type)

            # Log success
            logger.info(f"Successfully retrieved {len(document_results)} results for query: '{query_text}'")

            # Update cache with this result
            self.query_cache[cache_key] = (formatted_context, document_results)
            
            # Maintain cache size limit
            if len(self.query_cache) > self.max_cache_size:
                # Remove oldest entry (first key)
                oldest_key = next(iter(self.query_cache))
                self.query_cache.pop(oldest_key)

            # Return both the formatted context and the document results
            return formatted_context, document_results

        except Exception as e:
            logger.error(f"Error in _process_single_query: {str(e)}", exc_info=True)
            error_message = f"Error retrieving information: {str(e)}"
            return error_message, []
            
    def _format_context(self, results: List[Dict], expected_result_type: str = "factual") -> str:
        """
        Format the retrieved results into a readable context string.

        This method organizes search results into a structured format optimized for the
        expected result type. It adapts the presentation based on whether the query
        requires factual information, explanations, or summaries.

        Args:
            results: List of document dictionaries, each containing:
                - id: Document identifier
                - text: Document content
                - metadata: Additional document information
            expected_result_type: The type of result expected, which determines formatting:
                - "factual": Standard document presentation with source details
                - "conceptual": Content-focused presentation for explanatory material
                - "procedural": Step-by-step focused presentation
                - "opinion": Perspective-focused presentation

        Returns:
            A formatted string containing all relevant document contents, organized
            in a way that's appropriate for the expected result type.
        """
        if not results:
            return "No relevant information about this query was found in the knowledge base."

        # Format each result
        sections = []
        
        # Check if we need to add a no-specific-info warning at the beginning
        add_general_warning = False
        requested_db_type = None
        has_db_specific_info = False
        
        # Detect if there's a specific database requested but no matching info
        for result in results:
            metadata = result.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
                    
            # Extract database type
            db_type = metadata.get("database_type", metadata.get("document_subject", ""))
            if db_type and "warning" in metadata:
                # Found a warning which likely means a database mismatch
                requested_db_type = metadata.get("warning", "").split("for ")[-1].split(".")[0]
            elif db_type:
                # Check if we have at least one document matching the requested db
                if requested_db_type and db_type.lower() == requested_db_type.lower():
                    has_db_specific_info = True

        # Add general warning if requested DB info wasn't found
        if requested_db_type and not has_db_specific_info:
            add_general_warning = True
            
        # Add a header warning if needed
        if add_general_warning:
            warning = (f"⚠️ **Note**: No specific information for {requested_db_type} was found in the knowledge base. "
                      f"The following information may refer to other database systems and may not be directly applicable.\n\n")
            sections.append(warning)

        for i, result in enumerate(results):
            # Format the text content
            content = result["text"]
            
            # Get source info from metadata
            metadata = result.get("metadata", {})
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
                    
            # Extract useful metadata
            connection_type = metadata.get("connection_type", metadata.get("document_type", "")).upper()
            database_type = metadata.get("database_type", metadata.get("document_subject", "")).capitalize()
            content_category = metadata.get("content_category", "").capitalize()
            source_document = metadata.get("source_document", metadata.get("source", ""))
            warning = metadata.get("warning", "")
            
            # Create header with metadata
            header = f"**DOCUMENT {i+1}**"
            
            # Add source information when available
            if connection_type and database_type:
                header = f"**{connection_type}"
                if database_type != "General":
                    header += f" - {database_type}"
                if content_category and content_category != "General":
                    header += f" ({content_category})"
                header += "**"
            
            # Add warning for database-specific content that doesn't match requested database
            warning_text = ""
            if warning:
                warning_text = f"\n\n⚠️ *{warning}*"
            
            # Format based on expected_result_type, aligned with the query model types
            if expected_result_type == "conceptual":
                # For conceptual/explanatory content
                section = f"{header}\n\n{content}{warning_text}\n\n"

            elif expected_result_type == "procedural":
                # For procedural content with steps
                section = f"{header}\n\n{content}{warning_text}\n\n"
                
            elif expected_result_type == "opinion":
                # For opinion-based content
                section = f"{header}\n\n{content}{warning_text}\n\n"

            else:
                # Default factual format
                section = f"{header}\n\n{content}{warning_text}\n\n"
                
            # Add source document reference if available
            if source_document:
                section += f"*Source: {source_document}*\n\n"

            sections.append(section)
        
        # Combine everything - no header about search methods
        formatted_context = "\n".join(sections)
        
        return formatted_context

    def retrieve_with_results(self, retrieval_query: RetrievalQuery) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Retrieve information and return both formatted context and raw results.

        This is an internal method used for the fallback mechanism in the RetrieverAgent.
        It performs the same search as retrieve_information but returns both the
        formatted context string and the raw document results.

        Args:
            retrieval_query: The structured query object

        Returns:
            A tuple containing:
                - formatted_context: The formatted context string
                - results: The raw document results list
        """
        try:
            # The VectorStore returns a list of document dictionaries
            document_results = self.vector_store.query(
                query_text=retrieval_query.query,
                n_results=retrieval_query.n_results,
                metadata_filter=retrieval_query.metadata_filter,
                search_type=retrieval_query.search_type,
            )

            # Check if we have any results
            if not document_results or len(document_results) == 0:
                return "No information related to your query was found in the knowledge base.", []

            # Format the context
            formatted_context = self._format_context(document_results, retrieval_query.expected_result_type)

            return formatted_context, document_results

        except Exception as e:
            logger.error(f"Error in retrieve_with_results: {str(e)}", exc_info=True)
            return f"Error retrieving information: {str(e)}", []

    def direct_query(self, query_text: str, n_results: int = 5, metadata_filter: Optional[Dict[str, Any]] = None, 
                search_type: str = "vector") -> List[Dict[str, Any]]:
        """
        Perform a direct query to the vector store, bypassing agent logic.
        
        This is used as a fallback mechanism when the standard retrieve method fails.
        
        Args:
            query_text: The query text to search for
            n_results: Number of results to return
            metadata_filter: Optional filter for metadata fields
            search_type: Type of search to use (defaults to "vector" for reliability)
            
        Returns:
            List of document dictionaries with text and metadata
        """
        try:
            logger.info(f"Performing direct query: '{query_text}', search_type={search_type}")
            
            # Query the vector store directly
            results = self.vector_store.query(
                query_text=query_text,
                n_results=n_results,
                metadata_filter=metadata_filter,
                search_type=search_type
            )
            
            logger.info(f"Direct query found {len(results) if results else 0} results")
            return results
        except Exception as e:
            logger.error(f"Error in direct query: {str(e)}", exc_info=True)
            return []

class RetrieverAgent:
    """
    Agent responsible for retrieving information from the knowledge base using intelligent reasoning.

    This agent:
    1. Analyzes and refines user queries to optimize search effectiveness
    2. Makes strategic decisions about search methods based on query characteristics
    3. Searches the vector database using appropriate search strategies (hybrid, vector, text)
    4. Applies reranking to improve result quality and relevance
    5. Evaluates search results for relevance and completeness
    6. Formats information in a way that's optimized for the query type
    7. Returns contextually appropriate results with proper organization

    The agent uses LLM reasoning to enhance the retrieval process, making it more
    intelligent than a simple search function. It can adapt its approach based on
    the query complexity, expected result type, and initial search results.
    """

    def __init__(self,
                 model_id: str = Config.MODEL_ID,
                 vector_store: VectorStore = None):
        """
        Initialize the Retriever Agent.

        Args:
            model_id: ID of the OpenAI model to use
            vector_store: Vector store for querying documents
        """
        # Set up agent config
        self.model_id = model_id

        # Initialize vector store connection
        self.vector_store = vector_store

        # Create abilities for the agent
        self.info_retrieval_toolkit = InfoRetrievalToolkit(
            vector_store=vector_store
        )

        # Initialize the agent with specialized instructions
        self.agent = Agent(
            name="Retriever Agent",
            model=OpenAIChat(id=model_id, api_key=Config.OPENAI_API_KEY),
            description="Specialized agent for retrieving relevant information from the knowledge base",
            instructions=[
                "You are an expert information retrieval specialist responsible for finding the most relevant information from a knowledge base.",

                "Step 1: Query Analysis",
                "- Analyze the user's query to understand information needs",
                "- Identify key concepts and entities in the query",
                "- Expand, clarify, or decompose complex queries",
                "- Determine the appropriate search strategy based on query type",
                "- Consider expected result types (factual, conceptual, etc.) when planning retrieval",

                "Step 2: Information Retrieval",
                "- Search the vector database using available search methods (dense, sparse, hybrid)",
                "- Use the retrieve_information ability to execute searches",
                "- Evaluate result quality and relevance to the original query",
                "- If initial results are insufficient, try alternative search approaches",
                "- Apply metadata filters when appropriate to target specific document types",

                "Step 3: Result Evaluation",
                "- Judge the relevance and completeness of retrieved information",
                "- Determine if results directly address the user's information needs",
                "- Ensure diverse perspectives are included when appropriate",
                "- Structure information for optimal understanding",
                "- Prepare results in the format most appropriate for the expected result type",

                "Important Guidelines:",
                "- Use retrieve_information ability to actually perform searches",
                "- Make decisions about the best search strategy based on query characteristics",
                "- Always return a RetrievalResult structure with your findings",
                "- Adapt your approach based on the query complexity and domain",
                "- Report when relevant information cannot be found",
                "- Never fabricate or modify the content of retrieved documents"
            ],
            tools=[self.info_retrieval_toolkit],
            structured_outputs=True,
            markdown=True,
            debug_mode=True,
            show_tool_calls=True
        )

    def retrieve(self, retrieval_query: RetrievalQuery) -> RetrievalResult:
        """
        Retrieve relevant information from the knowledge base using agent-based reasoning.

        This method leverages LLM reasoning to:
        1. Analyze and understand the query intent and requirements
        2. Determine the optimal search strategy based on query characteristics
        3. Execute the search using the appropriate method and parameters
        4. Evaluate the quality and relevance of retrieved results
        5. Format the results in a way that best addresses the query

        The agent makes strategic decisions throughout the retrieval process,
        potentially trying different approaches if initial results are insufficient.

        Args:
            retrieval_query_or_dict: Either:
                - A RetrievalQuery object, or
                - A dictionary containing retrieval parameters:
                  - query: The text to search for
                  - n_results: Number of results to retrieve
                  - metadata_filter: Optional filters to apply to search results
                  - search_type: Search strategy to use (hybrid, vector, text)
                  - expected_result_type: Type of result expected
                  - query_context: Additional context about the query

        Returns:
            RetrievalResult object containing:
                - success: Whether the retrieval was successful
                - query: The original query
                - results: List of retrieved document dictionaries
                - context: Formatted context from the results
                - error_message: Error message if retrieval failed

        Note:
            This method uses LLM reasoning to enhance the retrieval process, making it
            more intelligent than a simple search function. If the agent fails to return
            a structured result, a fallback mechanism will execute a direct search.
        """
        try:
            logger.info(f"Agent retrieving information for: '{retrieval_query.query}'")

            try:
                """ # Create a prompt for the agent to reason about the retrieval
                prompt = f"""
                """ I need to find information related to this query: "{retrieval_query.query}"

                {f"QUERY CONTEXT:\n{retrieval_query.query_context}\n" if retrieval_query.query_context else ""}

                TASK 1: ANALYZE AND REFINE THE QUERY
                - Analyze the query to understand what information is being requested
                - Expand abbreviations and ambiguous terms if necessary
                - Identify key concepts and entities in the query

                For example, if the query is about "DD Oracle SSL config", understand that:
                - "DD" refers to DataDirect
                - "SSL config" refers to SSL/TLS security configuration
                - The query is about configuring SSL security in DataDirect Oracle drivers

                TASK 2: DETERMINE SEARCH STRATEGY
                - Based on the query, determine the best search strategy
                - Search types available: {retrieval_query.search_type}
                - Consider the expected result type: {retrieval_query.expected_result_type}

                For DataDirect documentation queries:
                - Use hybrid search for configuration and troubleshooting questions
                - Consider metadata filters for specific drivers (Oracle, SQL Server, etc.)
                - For procedural queries, prioritize step-by-step instructions

                TASK 3: RETRIEVE AND EVALUATE RESULTS
                - Use the InfoRetrievalToolkit toolkit to search for relevant documents
                - Evaluate if the results are satisfactory and relevant
                - If results seem insufficient, consider adjusting the search parameters

                Number of results requested: {retrieval_query.n_results}
                Metadata filters: {retrieval_query.metadata_filter if retrieval_query.metadata_filter else "None"}

                You have access to the 'InfoRetrievalToolkit' toolkit which queries the vector store. """
                """

                # Run the agent to handle the query
                agent_result = self.agent.run(prompt)

                # The agent should have used the toolkit's retrieve_information method
                # We need to extract the results for our return value

                # Check if agent result contains a RetrievalResult - safely access content attribute
                agent_content = getattr(agent_result, 'content', None)
                if isinstance(agent_content, RetrievalResult):
                    return agent_content

                # If we don't get a structured result, we'll need to use a direct fallback approach
                logger.warning("Agent did not return a structured RetrievalResult, using fallback method") """

                # Use the toolkit's direct retrieval capability instead of calling vector_store twice
                try:
                    # Direct search without agent reasoning
                    context, document_results = self.info_retrieval_toolkit.retrieve_information(retrieval_query)

                    if not document_results or len(document_results) == 0:
                        return RetrievalResult(
                            success=False,
                            query=retrieval_query.query,
                            results=[],
                            context="No information related to your query was found in the knowledge base.",
                            error_message="No results found"
                        )

                    return RetrievalResult(
                        success=True,
                        query=retrieval_query.query,
                        results=document_results[:retrieval_query.n_results],
                        context=context
                    )
                except Exception as e:
                    logger.error(f"Fallback retrieval failed: {str(e)}", exc_info=True)
                    return RetrievalResult(
                        success=False,
                        query=retrieval_query.query,
                        results=[],
                        context="",
                        error_message=f"Fallback retrieval failed: {str(e)}"
                    )

            except Exception as e:
                logger.error(f"Error retrieving information: {str(e)}", exc_info=True)
                return RetrievalResult(
                    success=False,
                    query=retrieval_query.query,
                    results=[],
                    context="",
                    error_message=f"Error retrieving information: {str(e)}"
                )

        except Exception as e:
            logger.error(f"Error in retrieve method: {str(e)}", exc_info=True)
            # Get the query from the input if possible, otherwise use "unknown"
            query = "unknown"
            if isinstance(retrieval_query, RetrievalQuery):
                query = retrieval_query.query
                
            return RetrievalResult(
                success=False,
                query=query,
                results=[],
                context="",
                error_message=f"Error in retrieve method: {str(e)}"
            )

# For backward compatibility
InfoRetrievalToolkit = InfoRetrievalToolkit

@lru_cache(maxsize=50)
def get_common_patterns(key: str) -> List[str]:
    """
    Cache common patterns used in queries to improve performance.
    This is useful for frequently accessed patterns in keyword extraction.
    
    Args:
        key: A unique key for the pattern type
        
    Returns:
        List of pattern strings
    """
    if key == "connection_types":
        return ["jdbc", "odbc", "hdp"]
    elif key == "database_types":
        return ["sqlserver", "oracle", "redshift", "postgres", "mysql"]
    elif key == "content_categories":
        return ["connection", "authentication", "querying", "configuration", "performance", "troubleshooting"]
    return []