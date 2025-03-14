import os
import json
import hashlib
import chromadb
import numpy as np  # Add numpy import
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from typing import List, Dict, Any, Optional, Union, Tuple
from agno.embedder.openai import OpenAIEmbedder
from chromadb.errors import InvalidCollectionException
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VectorStore")

class CachedEmbeddingFunction:
    """A wrapper that adds caching to any embedding function"""
    
    def __init__(self, embedding_function, cache_dir: str = "./.cache/embeddings"):
        """
        Initialize the cached embedding function.
        
        Args:
            embedding_function: The embedding function to wrap
            cache_dir: Directory to store cached embeddings
        """
        self.embedding_function = embedding_function
        self.cache_dir = cache_dir
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing cache index if it exists
        self.cache_index_path = os.path.join(cache_dir, "cache_index.json")
        if os.path.exists(self.cache_index_path):
            try:
                with open(self.cache_index_path, 'r') as f:
                    self.cache_index = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache index: {str(e)}. Creating new index.")
                self.cache_index = {}
        else:
            self.cache_index = {}
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for text with caching.
        
        Args:
            input: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        results = []
        texts_to_embed = []
        original_indices = []
        
        # Check cache for each input text
        for i, text in enumerate(input):
            # Create a hash of the text for cache key
            text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
            
            # Check if in cache
            if text_hash in self.cache_index:
                cache_file = os.path.join(self.cache_dir, f"{text_hash}.json")
                try:
                    with open(cache_file, 'r') as f:
                        cached_embedding = json.load(f)
                        results.append(cached_embedding)
                        self.cache_hits += 1
                        continue
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding: {str(e)}")
                    # Fall back to computing embedding
            
            # If not in cache, add to list to compute
            texts_to_embed.append(text)
            original_indices.append(i)
            self.cache_misses += 1
        
        # If there are texts to embed, compute them
        if texts_to_embed:
            new_embeddings = self.embedding_function(texts_to_embed)
            
            # Store new embeddings in cache
            for j, embedding in enumerate(new_embeddings):
                text = texts_to_embed[j]
                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                
                # Convert NumPy arrays to Python lists for JSON serialization
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                # Save to cache file
                cache_file = os.path.join(self.cache_dir, f"{text_hash}.json")
                with open(cache_file, 'w') as f:
                    json.dump(embedding, f)
                
                # Update cache index
                self.cache_index[text_hash] = True
                
                # Save updated index periodically
                if (self.cache_hits + self.cache_misses) % 100 == 0:
                    self._save_cache_index()
        
        # Merge cached and new embeddings
        final_results = [None] * len(input)
        
        # Place cached results
        for i, emb in enumerate(results):
            final_results[i] = emb
        
        # Place new results
        if texts_to_embed:
            for j, emb in enumerate(new_embeddings):
                # Convert NumPy arrays to Python lists for JSON serialization
                if isinstance(emb, np.ndarray):
                    emb = emb.tolist()
                idx = original_indices[j]
                final_results[idx] = emb
        
        # Log cache stats periodically
        if (self.cache_hits + self.cache_misses) % 100 == 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses) * 100 if (self.cache_hits + self.cache_misses) > 0 else 0
            logger.info(f"Embedding cache stats: {self.cache_hits} hits, {self.cache_misses} misses ({hit_rate:.1f}% hit rate)")
        
        return final_results
    
    def _save_cache_index(self):
        """Save the cache index to disk."""
        try:
            with open(self.cache_index_path, 'w') as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            logger.warning(f"Failed to save cache index: {str(e)}")

class AgnoEmbeddingFunction:
    """Wrapper to make Agno's embedders compatible with ChromaDB's embedding function interface."""
    
    def __init__(self, embedder=None, id="text-embedding-3-small"):
        """
        Initialize the embedding function with an Agno embedder.
        
        Args:
            embedder: An existing Agno embedder instance, or None to create a new one
            id: The model ID to use if creating a new embedder
        """
        if embedder:
            self.embedder = embedder
        else:
            # Initialize with proper model ID
            logger.info(f"Initializing new OpenAIEmbedder with model ID: {id}")
            self.embedder = OpenAIEmbedder(id=id)
            # Verify that we have access to the embedder methods
            if hasattr(self.embedder, 'get_embedding'):
                logger.info(f"OpenAIEmbedder has get_embedding method: {self.embedder.get_embedding}")
            else:
                logger.warning(f"OpenAIEmbedder DOES NOT have get_embedding method")
            logger.info(f"OpenAIEmbedder initialized successfully: {type(self.embedder).__name__}")
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text documents.
        
        Args:
            input: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        # Log details about what we're trying to embed
        logger.info(f"Generating embeddings for {len(input)} text items")
        
        # Process each text individually using get_embedding
        embeddings = []
        for i, text in enumerate(input):
            try:
                logger.debug(f"Embedding text item {i+1}/{len(input)} (length: {len(text)})")
                # Following Agno's pattern of using get_embedding for both single and multiple texts
                embedding = self.embedder.get_embedding(text)
                
                # Convert numpy arrays to lists if needed
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()
                
                # Log embedding dimensionality
                if i == 0:  # Only log this for the first item to avoid flooding logs
                    logger.info(f"Successfully generated embedding with dim: {len(embedding)}")
                
                embeddings.append(embedding)
            except Exception as e:
                error_msg = f"Failed to generate embedding for text item {i+1}: {str(e)}"
                logger.error(error_msg)
                logger.error(f"Exception type: {type(e).__name__}")
                logger.error(f"Exception traceback: {traceback.format_exc()}")
                logger.error(f"Text length: {len(text)}")
                logger.error(f"Embedder type: {type(self.embedder).__name__}")
                logger.error(f"Full exception details: {str(e)}")
                raise ValueError(error_msg)
        
        logger.info(f"Successfully generated embeddings for all {len(input)} text items")
        return embeddings

class VectorStore:
    """Utility class for managing the ChromaDB vector store."""
    
    def __init__(self, 
                 persistence_directory: str = "./data/vector_store",
                 collection_name: str = "document_store",
                 embedding_function: Optional[Any] = None,
                 enable_caching: bool = True,
                 cache_dir: str = "./.cache/embeddings",
                 batch_size: int = 100):
        """
        Initialize the vector store with ChromaDB.
        
        Args:
            persistence_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_function: Optional custom embedding function
            enable_caching: Whether to enable embedding caching
            cache_dir: Directory to store cached embeddings
            batch_size: Batch size for adding documents
        """
        # Ensure the persistence directory exists
        os.makedirs(persistence_directory, exist_ok=True)
        
        # Store batch size
        self.batch_size = batch_size
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persistence_directory,
            settings=Settings(
                anonymized_telemetry=False,  # Disable telemetry
                allow_reset=True,  # Allow collection reset if needed
            )
        )
        
        # Create or get the embedding function
        if embedding_function is None:
            # Use ChromaDB's built-in OpenAI embedding function
            # This will automatically get the API key from the environment variable OPENAI_API_KEY
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            raw_embedding_function = OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name="text-embedding-3-small"
            )
        else:
            raw_embedding_function = embedding_function
            
        # Apply caching if enabled
        if enable_caching:
            self.embedding_function = CachedEmbeddingFunction(
                raw_embedding_function,
                cache_dir=cache_dir
            )
        else:
            self.embedding_function = raw_embedding_function
            
        # Initialize collection
        self.collection = self._get_or_create_collection(collection_name)
        
        # Initialize query cache
        self._query_cache = {}
        self._max_query_cache_size = 100
    
    def _get_or_create_collection(self, collection_name: str) -> Collection:
        """Get or create a ChromaDB collection."""
        try:
            return self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
        except InvalidCollectionException:
            return self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'id', 'text', and 'metadata'
        """
        if not documents:
            logger.warning("No documents to add to vector store")
            return
            
        logger.info(f"Adding {len(documents)} documents to vector store")
        
        # Extract IDs, texts, and metadata from documents
        ids = [doc["id"] for doc in documents]
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        
        # Add to collection in batches to avoid memory issues
        added_count = 0
        for i in range(0, len(ids), self.batch_size):
            batch_ids = ids[i:i+self.batch_size]
            batch_texts = texts[i:i+self.batch_size]
            batch_metadatas = metadatas[i:i+self.batch_size]
            
            # Log progress
            batch_end = min(i + self.batch_size, len(ids))
            logger.info(f"Adding batch {i+1}-{batch_end} of {len(ids)}")
            
            # Add the batch to the collection
            try:
                self.collection.add(
                    ids=batch_ids,
                    documents=batch_texts,
                    metadatas=batch_metadatas
                )
                added_count += len(batch_ids)
            except Exception as e:
                logger.error(f"Error adding batch to vector store: {str(e)}")
        
        logger.info(f"Successfully added {added_count} documents to vector store")
        
        # Clear query cache after adding new documents
        self._query_cache = {}
    
    def query(self, 
              query_text: str, 
              n_results: int = 5, 
              metadata_filter: Optional[Dict] = None,
              use_cache: bool = True,
              min_relevance_score: float = 0.75) -> Dict:
        """
        Query the vector store for documents similar to the query text.
        
        Args:
            query_text: Text to query
            n_results: Maximum number of results to return
            metadata_filter: Optional filter to apply to metadata
            use_cache: Whether to use the query cache
            min_relevance_score: Minimum relevance score (1.0 - distance) for results
            
        Returns:
            Dictionary containing query results
        """
        # Generate cache key
        if use_cache:
            # Convert metadata_filter to a JSON string if it exists
            metadata_filter_str = json.dumps(metadata_filter) if metadata_filter else 'None'
            cache_key = f"{query_text}_{n_results}_{min_relevance_score}_{metadata_filter_str}"
            
            # Check cache
            if cache_key in self._query_cache:
                logger.info(f"Using cached results for query: {query_text[:30]}...")
                return self._query_cache[cache_key]
        
        # Build the query
        try:
            # Request more results than needed to account for filtering
            actual_n_results = min(n_results * 3, 20)  # Get more results for filtering but cap at 20
            
            # Execute query with proper error handling
            try:
                if metadata_filter:
                    results = self.collection.query(
                        query_texts=[query_text],
                        n_results=actual_n_results,
                        where=metadata_filter
                    )
                else:
                    results = self.collection.query(
                        query_texts=[query_text],
                        n_results=actual_n_results
                    )
            except ValueError as ve:
                if "embeddings" in str(ve).lower():
                    logger.error(f"Embedding error during query: {str(ve)}")
                    return {
                        "ids": [[]],
                        "documents": [[]],
                        "metadatas": [[]],
                        "distances": [[]]
                    }
                else:
                    # Re-raise other ValueError exceptions
                    raise
            
            # Cache results if enabled
            if use_cache:
                # Make sure the results are JSON serializable
                serializable_results = self._make_serializable(results)
                
                # Limit cache size with LRU policy
                if len(self._query_cache) >= self._max_query_cache_size:
                    # Remove oldest item (first key)
                    oldest_key = next(iter(self._query_cache))
                    del self._query_cache[oldest_key]
                
                # Add to cache
                self._query_cache[cache_key] = serializable_results
                
                return serializable_results
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            # Return empty results on error
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }
    
    def _make_serializable(self, results: Dict) -> Dict:
        """
        Convert query results to a JSON serializable format.
        
        Args:
            results: Query results from ChromaDB
            
        Returns:
            JSON serializable results dictionary
        """
        serializable_results = {}
        
        # Copy the structure but ensure all elements are serializable
        for key, value in results.items():
            if key == "distances" and value:
                # Convert numpy arrays to lists if needed
                if isinstance(value[0], np.ndarray):
                    serializable_results[key] = [arr.tolist() for arr in value]
                else:
                    serializable_results[key] = value
            else:
                serializable_results[key] = value
        
        return serializable_results 