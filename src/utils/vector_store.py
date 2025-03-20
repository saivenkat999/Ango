import os
import json
import hashlib
from functools import lru_cache
from typing import List, Dict, Any, Optional, Union, Iterable, Tuple
import logging
import lancedb
import numpy as np
import pyarrow as pa
from src.utils.config import Config
import pandas as pd
from lancedb.pydantic import LanceModel, Vector
from lancedb.rerankers import LinearCombinationReranker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VectorStore")

class VectorStore:
    """Utility class for managing the LanceDB vector store with hybrid search capabilities."""

    def __init__(self,
                 persistence_directory: str = None,
                 collection_name: str = None,
                 embedding_model_id: str = None,
                 search_type: str = None,
                 batch_size: int = None):
        """
        Initialize the vector store with LanceDB.

        Args:
            persistence_directory: Directory to persist LanceDB data (defaults to Config.PERSISTENCE_DIRECTORY)
            collection_name: Name of the LanceDB table (defaults to Config.COLLECTION_NAME)
            embedding_model_id: ID of the embedding model to use (defaults to Config.EMBEDDING_MODEL_ID)
            search_type: Type of search to use (defaults to Config.SEARCH_TYPE)
            batch_size: Batch size for adding documents (defaults to Config.BATCH_SIZE)
        """
        # Use provided values or defaults from Config
        self.persistence_directory = persistence_directory or Config.PERSISTENCE_DIRECTORY
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.embedding_model_id = embedding_model_id or Config.EMBEDDING_MODEL_ID
        self.search_type = search_type or Config.SEARCH_TYPE
        self.batch_size = batch_size or Config.BATCH_SIZE
        
        # Ensure the persistence directory exists
        os.makedirs(self.persistence_directory, exist_ok=True)

        # Initialize FastEmbed model
        logger.info(f"Initializing FastEmbedder with model ID: {self.embedding_model_id}")
        try:
            from fastembed import TextEmbedding
            
            # Check if the model ID is valid
            available_models = TextEmbedding.list_supported_models()
            model_ids = [model.get('model', '').lower() for model in available_models]
            
            # Log available models for debugging
            logger.debug(f"Available embedding models: {model_ids}")
            
            # Check if the specified model is available
            if self.embedding_model_id.lower() not in [m.lower() for m in model_ids]:
                logger.warning(f"Model ID '{self.embedding_model_id}' not found in supported models. Using default model.")
                # Use a default model if the specified one is not available
                self.embedder = TextEmbedding()
                self.embedding_model_id = "BAAI/bge-small-en"  # Default model
            else:
                self.embedder = TextEmbedding(model_name=self.embedding_model_id)
            
            # Get dimensions by embedding a test string
            test_embedding = next(self.embedder.embed("test"))
            self.embedding_dim = len(test_embedding)
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
            # Initialize the reranker for hybrid search based on configuration
            reranker_type = Config.RERANKER_TYPE.lower()
            if reranker_type == "linear":
                from lancedb.rerankers import LinearCombinationReranker
                self.reranker = LinearCombinationReranker(weight=Config.RERANKER_WEIGHT)
                logger.info(f"Initialized LinearCombinationReranker with weight {Config.RERANKER_WEIGHT}")
            elif reranker_type == "cohere":
                try:
                    from lancedb.rerankers import CohereReranker
                    self.reranker = CohereReranker(
                        model_name=Config.RERANKER_MODEL,
                        api_key=Config.COHERE_API_KEY
                    )
                    logger.info(f"Initialized CohereReranker with model {Config.RERANKER_MODEL}")
                except ImportError:
                    logger.warning("Cohere reranker not available. Falling back to LinearCombinationReranker")
                    from lancedb.rerankers import LinearCombinationReranker
                    self.reranker = LinearCombinationReranker(weight=Config.RERANKER_WEIGHT)
            elif reranker_type == "rrf":
                from lancedb.rerankers import RRFReranker
                self.reranker = RRFReranker()
                logger.info("Initialized RRFReranker (Reciprocal Rank Fusion)")
            else:
                # Default fallback
                from lancedb.rerankers import LinearCombinationReranker
                self.reranker = LinearCombinationReranker(weight=0.7)
                logger.info(f"Unknown reranker type '{reranker_type}', using LinearCombinationReranker")
            
            # Initialize cache for embeddings
            self._embedding_cache = {}
            
        except ImportError:
            logger.error("fastembed not installed. Please install with 'pip install fastembed'")
            raise
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise

        # Initialize LanceDB connection
        self.db = lancedb.connect(self.persistence_directory)

        # Check if table exists and drop if necessary for testing purposes
        if self.collection_name in self.db.table_names():
            logger.info(f"Using existing table: {self.collection_name}")
            self.table = self.db.open_table(self.collection_name)
            
            # Check if vector index exists, create if it doesn't
            try:
                indices = list(self.table.list_indices())
                has_vector_index = any(idx.column == "vector" for idx in indices)
                if not has_vector_index:
                    logger.info("Creating missing vector index")
                    self.table.create_index(
                        column="vector",
                        index_type="IVF_PQ",
                        metric="L2"
                    )
            except Exception as e:
                logger.warning(f"Error checking indices: {str(e)}")
                
            # Create FTS index if needed and doesn't exist
            if self.search_type in ["hybrid", "text"]:
                try:
                    has_fts_index = any(idx.column == "text" and idx.index_type == "FTS" for idx in indices)
                    if not has_fts_index:
                        logger.info("Creating missing FTS index")
                        self.table.create_fts_index("text", replace=True)
                except Exception as e:
                    logger.warning(f"Error checking FTS index: {str(e)}")
        else:
            logger.info(f"Creating new table: {self.collection_name}")
            self._create_table()

        logger.info(f"Created LanceDB table: {self.collection_name} with correct schema")

    def _create_table(self):
        """Create the LanceDB table with the proper schema."""
        # Define the schema using PyArrow
        schema = pa.schema([
            pa.field("vector", lancedb.vector(self.embedding_dim)),  # Vector type for embeddings
            pa.field("id", pa.string()),  # String type for ID
            pa.field("text", pa.string()),  # String type for text
            pa.field("metadata", pa.string())  # Store metadata as serialized JSON
        ])

        # Create the table with the schema
        self.table = self.db.create_table(
            name=self.collection_name,
            schema=schema,
            mode="overwrite"
        )

        # Create full-text search index if needed for hybrid or text search
        if self.search_type in ["hybrid", "text"]:
            self.table.create_fts_index("text", replace=True)
        
        # Create vector index for vector search
        self.table.create_index(
            column="vector",
            index_type="IVF_PQ",
            metric="L2",
            replace=True
        )

    @lru_cache(maxsize=100)
    def _get_embedding_cached(self, text_hash: str) -> np.ndarray:
        """
        Get cached embedding for text hash or generate if not in cache.
        
        Args:
            text_hash: Hash of the text to get embedding for
            
        Returns:
            The embedding vector
        """
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        return None
        
    def _embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for text using the configured embedder."""
        # Create a hash of the text to use as cache key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Check if we have a cached embedding
        cached_embedding = self._get_embedding_cached(text_hash)
        if cached_embedding is not None:
            return cached_embedding
            
        try:
            # The embedder.embed() returns an iterator of numpy arrays
            # We take the first (and only) embedding
            embedding = next(self.embedder.embed(text))
            
            # Cache the embedding
            self._embedding_cache[text_hash] = embedding
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Return zeros array if embedding fails
            return np.zeros(self.embedding_dim)

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document dictionaries with text and metadata
        """
        if not documents:
            logger.warning("No documents to add")
            return

        # Convert documents to LanceDB format
        lance_docs = []

        for doc in documents:
            # Skip documents without text
            if not doc.get('text'):
                logger.warning(f"Skipping document without text: {doc.get('id', 'NO_ID')}")
                continue

            # Generate document ID if not provided
            doc_id = str(doc.get('id', '')) or str(hash(doc['text']))

            # Generate embedding
            embedding = self._embed_text(doc['text'])

            # Create LanceDB document with serialized metadata
            lance_doc = {
                "id": doc_id,
                "vector": embedding,
                "text": doc['text'],
                "metadata": json.dumps(doc.get('metadata', {}))  # Serialize metadata to JSON
            }

            lance_docs.append(lance_doc)

        # Process documents in batches for better performance
        for i in range(0, len(lance_docs), self.batch_size):
            batch = lance_docs[i:i + self.batch_size]
            if batch:
                try:
                    # Add documents to LanceDB
                    self.table.add(batch)
                    logger.info(f"Added batch of {len(batch)} documents")
                except Exception as e:
                    logger.error(f"Error adding documents: {str(e)}")
                    raise

    def query(self,
             query_text: str,
             n_results: int = None,
             metadata_filter: Optional[Dict[str, Any]] = None,
             search_type: str = None) -> List[Dict[str, Any]]:
        """
        Query the vector store for relevant documents.

        Args:
            query_text: The query text to search for
            n_results: Number of results to return (defaults to Config.MAX_RESULTS)
            metadata_filter: Optional filter for metadata fields
            search_type: Type of search to use (defaults to self.search_type)

        Returns:
            List of dictionaries containing document info and relevance scores
        """
        # Validate inputs
        if not query_text or not isinstance(query_text, str):
            logger.error(f"Invalid query_text provided: {query_text}")
            return []
            
        # Normalize query text
        query_text = query_text.strip()
        if not query_text:
            logger.error("Empty query text after normalization")
            return []
            
        # Use Config.MAX_RESULTS if n_results not provided
        n_results = n_results or Config.MAX_RESULTS
        search_type = search_type or self.search_type or Config.SEARCH_TYPE
        
        start_time = pd.Timestamp.now()
        logger.debug(f"Starting query: {query_text}, search_type={search_type}, n_results={n_results}")
        
        try:
            # Ensure a connection to the database
            if not hasattr(self, 'db') or not hasattr(self, 'table'):
                logger.warning("Database connection not initialized, reconnecting...")
                self.db = lancedb.connect(self.persistence_directory)
                if self.collection_name in self.db.table_names():
                    self.table = self.db.open_table(self.collection_name)
                else:
                    logger.error(f"Table {self.collection_name} does not exist")
                    return []
            
            # Generate embedding for the query text
            query_embedding = self._embed_text(query_text)
            if query_embedding is None or np.all(query_embedding == 0):
                logger.error("Failed to generate embedding for query")
                return []
            
            # Initialize search based on search type
            if search_type == "hybrid":
                # For hybrid search, we need to manually generate the embedding
                query_embedding = self._embed_text(query_text)
                logger.debug(f"Using hybrid search with query: {query_text}")
                try:
                    # First, try using the built-in hybrid search with proper vector and text queries
                    hybrid_search = self.table.search(
                        query_type="hybrid"
                    ).vector(query_embedding).text(query_text)
                    
                    # Apply the reranker with the correct method signature
                    search_query = hybrid_search.rerank(self.reranker)
                    results = search_query.limit(n_results * 3).to_pandas()  # Get more results for filtering
                    logger.debug(f"Built-in hybrid search found {len(results)} results before metadata filtering")
                except Exception as e:
                    logger.warning(f"Built-in hybrid search failed: {str(e)}. Using enhanced manual hybrid search.")
                    
                    # Enhanced manual hybrid approach - perform both searches and combine results
                    try:
                        # Perform vector search
                        vector_results = self.table.search(
                            query=query_embedding,
                            vector_column_name="vector",
                            query_type="vector"
                        ).limit(n_results * 3).to_pandas()  # Get more results for filtering
                        logger.debug(f"Vector search found {len(vector_results)} results")
                        
                        # Perform text search
                        text_results = self.table.search(
                            query=query_text,
                            query_type="fts",
                            fts_columns=["text"]
                        ).limit(n_results * 3).to_pandas()  # Get more results for filtering
                        logger.debug(f"Text search found {len(text_results)} results")
                        
                        # Combine results from both searches
                        if not vector_results.empty and not text_results.empty:
                            # Assign scores - normalize and combine for better ranking
                            # For vector results, higher score is better (closer to 1.0)
                            if '_distance' in vector_results.columns:
                                # Convert distance to similarity score (1.0 - distance)
                                # Add a small epsilon to avoid division by zero
                                epsilon = 1e-10
                                max_distance = vector_results['_distance'].max() + epsilon
                                # Normalize distances to [0, 1] range and convert to similarity
                                vector_results['_score'] = 1.0 - (vector_results['_distance'] / max_distance)
                            
                            # Normalize text search scores if they exist
                            if '_score' in text_results.columns:
                                max_score = text_results['_score'].max()
                                if max_score > 0:
                                    text_results['_score'] = text_results['_score'] / max_score
                            else:
                                # If no score column exists, add a default score
                                text_results['_score'] = 0.5
                            
                            # Combine with preference to documents found in both searches
                            all_results = pd.concat([vector_results, text_results])
                            # Count occurrences of each id to prioritize results found by both methods
                            id_counts = all_results['id'].value_counts().to_dict()
                            all_results['found_in_both'] = all_results['id'].map(id_counts)
                            
                            # Calculate combined score - prioritize items found in both searches
                            # and give higher weight to vector search (0.7) than text search (0.3)
                            def calculate_combined_score(row):
                                base_score = row.get('_score', 0.5)
                                boost = 1.0 if row['found_in_both'] > 1 else 0.0
                                return base_score + (boost * 0.2)  # Add 0.2 boost for items found in both
                            
                            all_results['combined_score'] = all_results.apply(calculate_combined_score, axis=1)
                            
                            # Sort by combined score, descending
                            all_results = all_results.sort_values(by='combined_score', ascending=False)
                            
                            # Remove duplicates, keeping the first occurrence (highest score)
                            results = all_results.drop_duplicates(subset=['id'])
                            logger.debug(f"Combined search found {len(results)} unique results")
                        elif not vector_results.empty:
                            results = vector_results
                            logger.debug(f"Using only vector search results: {len(results)}")
                        elif not text_results.empty:
                            results = text_results
                            logger.debug(f"Using only text search results: {len(results)}")
                        else:
                            logger.warning("Both vector and text search returned empty results")
                            results = pd.DataFrame()
                    except Exception as e2:
                        logger.warning(f"Enhanced manual hybrid search failed: {str(e2)}. Falling back to simple vector search.")
                        # Fall back to simple vector search
                        search_query = self.table.search(
                            query=query_embedding,
                            vector_column_name="vector",
                            query_type="vector"
                        )
                        results = search_query.limit(n_results * 3).to_pandas()  # Get more results for filtering
                        logger.debug(f"Fallback vector search found {len(results)} results")
            elif search_type == "vector":
                # For vector search, we manually generate the embedding
                query_embedding = self._embed_text(query_text)
                logger.debug(f"Using vector search with embedding dimension: {len(query_embedding)}")
                search_query = self.table.search(
                    query=query_embedding,
                    vector_column_name="vector",
                    query_type="vector"
                )
                results = search_query.limit(n_results).to_pandas()
            else:  # text search
                logger.debug(f"Using text search with query: {query_text}")
                search_query = self.table.search(
                    query=query_text,
                    query_type="fts",
                    fts_columns=["text"]
                )
                results = search_query.limit(n_results).to_pandas()

            # Apply metadata filtering after search, but with improvements
            if metadata_filter and not results.empty:
                logger.debug(f"Applying metadata filter: {metadata_filter}")
                logger.debug(f"Before filtering: {len(results)} results")
                
                # Store original results for fallback
                original_results = results.copy()
                filtered_results = []
                
                # Check if there are database or connection type specific filters
                requested_connection_type = metadata_filter.get("connection_type")
                requested_database_type = metadata_filter.get("database_type")
                have_specific_filter = requested_connection_type or requested_database_type
                
                # First pass: enhanced filtering that understands the hierarchical structure
                for _, row in results.iterrows():
                    try:
                        # Parse metadata as JSON
                        row_metadata = json.loads(row['metadata'])
                        
                        # Initialize scoring
                        match_score = 0
                        total_match_factors = 0
                        matches = {}
                        
                        # Handle hierarchical filtering with specific rules for connection and database types
                        if have_specific_filter:
                            # Connection type matching (level 1)
                            if requested_connection_type:
                                total_match_factors += 2  # Give more weight to connection type
                                conn_type = row_metadata.get("connection_type", row_metadata.get("document_type"))
                                if conn_type and conn_type.lower() == requested_connection_type.lower():
                                    match_score += 2
                                    matches["connection_type"] = True
                                elif not row_metadata.get("connection_specific", True):
                                    # Item marked as not connection-specific, so it applies broadly
                                    match_score += 1
                                    matches["connection_type"] = "partial"
                            
                            # Database type matching (level 2)
                            if requested_database_type:
                                total_match_factors += 2  # Give more weight to database type
                                db_type = row_metadata.get("database_type", row_metadata.get("document_subject"))
                                if db_type and db_type.lower() == requested_database_type.lower():
                                    match_score += 2
                                    matches["database_type"] = True
                                elif not row_metadata.get("database_specific", True):
                                    # Item marked as not database-specific, so it applies broadly
                                    match_score += 1
                                    matches["database_type"] = "partial"
                                # Check if explicitly marked as not applicable to this DB
                                not_applicable = row_metadata.get("not_applicable_to", [])
                                if isinstance(not_applicable, list) and requested_database_type.lower() in [x.lower() for x in not_applicable]:
                                    match_score = 0  # Explicitly exclude this result
                                    matches["database_type"] = False
                        
                        # Handle any other metadata filters
                        for key, value in metadata_filter.items():
                            # Skip connection and database type since we already handled them
                            if key in ["connection_type", "database_type"]:
                                continue
                                
                            total_match_factors += 1
                            
                            # Check if this key exists in metadata
                            if key in row_metadata:
                                # For string values, check for partial match
                                if isinstance(row_metadata[key], str) and isinstance(value, str):
                                    if value.lower() in row_metadata[key].lower():
                                        match_score += 1
                                        matches[key] = True
                                # For list values, check for any overlap
                                elif isinstance(row_metadata[key], list) and isinstance(value, (str, list, tuple)):
                                    if isinstance(value, (list, tuple)):
                                        if any(v.lower() in str(item).lower() for v in value for item in row_metadata[key]):
                                            match_score += 1
                                            matches[key] = True
                                    else:
                                        if any(value.lower() in str(item).lower() for item in row_metadata[key]):
                                            match_score += 1
                                            matches[key] = True
                                # For exact matches
                                elif row_metadata[key] == value:
                                    match_score += 1
                                    matches[key] = True
                                    
                                # Check for partial matches in keywords field
                                elif key == "keywords" and "keywords" in row_metadata and isinstance(row_metadata["keywords"], list):
                                    if isinstance(value, str) and any(value.lower() in k.lower() for k in row_metadata["keywords"]):
                                        match_score += 0.5  # Partial credit for keyword match
                                        matches[key] = "partial"
                            
                            # Special handling for content category
                            elif key == "content_category" and "content_category" in row_metadata:
                                if value.lower() == row_metadata["content_category"].lower():
                                    match_score += 1
                                    matches[key] = True
                        
                        # Calculate final score (normalized by importance factors)
                        if total_match_factors > 0:
                            final_match_score = match_score / total_match_factors
                            
                            # Add warning metadata for database-specific content that doesn't match the requested database
                            if requested_database_type and "database_type" in row_metadata and row_metadata.get("database_specific", True):
                                db_type = row_metadata.get("database_type")
                                if db_type and db_type.lower() != requested_database_type.lower():
                                    # Add a warning to metadata
                                    row_metadata["warning"] = f"Information is specific to {db_type}, not confirmed for {requested_database_type}"
                                    # Store updated metadata back in the row
                                    row['metadata'] = json.dumps(row_metadata)
                            
                            # Consider a match if score is positive
                            if match_score > 0:
                                row['match_score'] = final_match_score
                                row['match_details'] = json.dumps(matches)
                                filtered_results.append(row)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON in metadata: {row['metadata']}")
                        continue
                
                # Sort by match score if we have filtered results
                if filtered_results:
                    filtered_results = sorted(filtered_results, key=lambda x: x.get('match_score', 0), reverse=True)
                    results = pd.DataFrame(filtered_results)
                    logger.debug(f"After filtering: {len(filtered_results)} results with enhanced metadata filtering")
                else:
                    # No results match filter - return top semantic matches instead of nothing
                    logger.warning("No results match the metadata filter, returning top semantic matches instead")
                    
                    # If database-specific filter was requested but not found, add warning
                    if requested_database_type:
                        # Create a system message as the first result to inform that specific data wasn't found
                        warning_added = False
                        for i, row in original_results.iterrows():
                            try:
                                row_metadata = json.loads(row['metadata'])
                                row_metadata["warning"] = f"No specific information found for {requested_database_type}. Results shown may not apply directly."
                                original_results.at[i, 'metadata'] = json.dumps(row_metadata)
                                warning_added = True
                            except:
                                pass
                            # Only add warning to first result
                            if warning_added:
                                break
                    
                    results = original_results.head(n_results)
            
            # Final limit on results
            if not results.empty:
                results = results.head(n_results)

            logger.debug(f"Results: {results}")
            # Format results
            formatted_results = []
            for _, row in results.iterrows():
                try:
                    metadata = json.loads(row['metadata'])
                except (json.JSONDecodeError, TypeError):
                    logger.warning(f"Invalid JSON in metadata: {row['metadata']}")
                    metadata = {}
                    
                # Ensure all expected fields are present
                formatted_result = {
                    'id': row.get('id', f"doc_{len(formatted_results)}"),
                    'text': row.get('text', ''),
                    'metadata': metadata
                }
                
                # Add score if available
                if '_score' in row:
                    formatted_result['score'] = float(row['_score'])
                elif 'combined_score' in row:
                    formatted_result['score'] = float(row['combined_score'])
                elif '_distance' in row:
                    # Convert distance to similarity score
                    formatted_result['score'] = float(1.0 - row['_distance'])
                
                formatted_results.append(formatted_result)

            # Calculate and log performance metrics
            end_time = pd.Timestamp.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000
            logger.info(f"Query completed in {duration_ms:.2f}ms, found {len(formatted_results)} results")
            
            # Add performance metrics to the result if in debug mode
            if logging.getLogger().level == logging.DEBUG:
                metrics = {
                    'duration_ms': duration_ms,
                    'total_results': len(formatted_results),
                    'search_type': search_type
                }
                logger.debug(f"Query metrics: {metrics}")
            
            # IMPORTANT: Always return the formatted list of dictionaries, never the pandas DataFrame
            return formatted_results

        except Exception as e:
            logger.error(f"Error during query: {str(e)}", exc_info=True)
            # Return empty results on error
            return []

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            table_exists = self.collection_name in self.db.table_names()
            info = {
                'name': self.collection_name,
                'exists': table_exists,
                'count': self.table.count_rows() if table_exists else 0,
                'search_type': self.search_type
            }

            return info
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {
                'name': self.collection_name,
                'exists': False,
                'error': str(e)
            }

    def rerank_documents(self, query_text: str, documents: List[Dict[str, Any]], n_results: int = None) -> List[Dict[str, Any]]:
        """
        Rerank documents using the configured reranker.
        
        Args:
            query_text: The query text used for reranking
            documents: List of document dictionaries to rerank
            n_results: Number of results to return (default is Config.MAX_RESULTS)
            
        Returns:
            List of reranked document dictionaries
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []
            
        if not Config.USE_RERANKING:
            logger.info("Reranking is disabled, returning original documents")
            return documents[:n_results or Config.MAX_RESULTS]
        
        try:
            n_results = n_results or Config.MAX_RESULTS
            logger.info(f"Reranking {len(documents)} documents using {Config.RERANKER_TYPE} reranker")
            
            # Convert documents to a PyArrow table for LanceDB reranking
            import pyarrow as pa
            
            # Extract text and other fields from documents
            texts = [doc['text'] for doc in documents]
            ids = [doc.get('id', f"doc_{i}") for i, doc in enumerate(documents)]
            
            # Create a PyArrow table
            table_data = {
                'id': pa.array(ids),
                'text': pa.array(texts),
                # Include other fields needed for reranking
                'metadata': pa.array([json.dumps(doc.get('metadata', {})) for doc in documents])
            }
            
            # Add scores if they exist
            if 'score' in documents[0]:
                table_data['_score'] = pa.array([float(doc.get('score', 0.0)) for doc in documents])
                
            pa_table = pa.Table.from_pydict(table_data)
            
            # Apply reranking based on the reranker type
            if Config.RERANKER_TYPE.lower() == "cohere":
                # Use LanceDB's CohereReranker directly
                from lancedb.rerankers import CohereReranker
                
                reranker = CohereReranker(
                    model_name=Config.RERANKER_MODEL,
                    api_key=Config.COHERE_API_KEY,
                    return_score="all"  # This prevents dropping _distance column
                )
                reranked_table = reranker.rerank_vector(query_text, pa_table)
                
                # Convert the reranked table back to documents
                reranked_docs = []
                for i in range(min(len(reranked_table), n_results)):
                    row = reranked_table.slice(i, 1)
                    doc = {
                        'id': row['id'][0].as_py(),
                        'text': row['text'][0].as_py(),
                        'metadata': json.loads(row['metadata'][0].as_py())
                    }
                    
                    # Add relevance score if available
                    if '_relevance_score' in row.column_names:
                        doc['score'] = float(row['_relevance_score'][0].as_py())
                    elif '_score' in row.column_names:
                        doc['score'] = float(row['_score'][0].as_py())
                        
                    reranked_docs.append(doc)
                
                logger.info(f"Successfully reranked documents, returning {len(reranked_docs)} results")
                return reranked_docs
            
            elif Config.RERANKER_TYPE.lower() in ["linear", "rrf"]:
                # For these reranker types, we'll perform reranking at the LanceDB query level
                # This is already handled in the query method, so just return the top documents
                logger.info(f"Using {Config.RERANKER_TYPE} reranker which was applied during the query")
                return documents[:n_results]
            
            else:
                logger.warning(f"Unknown reranker type: {Config.RERANKER_TYPE}, returning original documents")
                return documents[:n_results]
                
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}", exc_info=True)
            # Fall back to returning the original documents
            return documents[:n_results or Config.MAX_RESULTS]

    def cleanup(self):
        """Remove the existing table and recreate it."""
        try:
            if self.collection_name in self.db.table_names():
                self.db.drop_table(self.collection_name)
                logger.info(f"Removed table '{self.collection_name}'")
                
            # Recreate the table with the proper schema
            self._create_table()
            logger.info(f"Recreated table '{self.collection_name}' with proper schema")
            
            return True
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
            return False