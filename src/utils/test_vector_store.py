import os
import sys
import logging
from typing import List, Dict, Any

# Add the src directory to the path so we can import the VectorStore
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.utils.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("VectorStoreTest")

def generate_sample_documents(num_docs: int = 5) -> List[Dict[str, Any]]:
    """Generate sample documents for testing."""
    documents = []

    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence that involves training algorithms on data.",
        "Vector stores are databases designed to store and search vector embeddings efficiently.",
        "Natural language processing enables computers to understand human language.",
        "The transformer architecture revolutionized the field of NLP with its attention mechanism.",
        "Deep learning models require large amounts of data to train effectively.",
        "Python is a popular programming language for data science and machine learning.",
        "LanceDB is a vector database that provides efficient vector search capabilities.",
        "Retrieval Augmented Generation combines information retrieval with text generation.",
        "Semantic search uses vector embeddings to find results based on meaning rather than keywords."
    ]

    for i in range(min(num_docs, len(sample_texts))):
        doc = {
            "id": f"doc_{i+1}",
            "text": sample_texts[i],
            "metadata": {
                "source": "test_data",
                "category": f"category_{(i % 3) + 1}",
                "importance": (i % 5) + 1
            }
        }
        documents.append(doc)

    return documents

def test_vector_store():
    """Test the VectorStore class functionality."""
    # Create a test directory for the vector store
    test_dir = os.path.join("test_data", "vector_store")

    try:
        # Initialize the vector store
        logger.info("Initializing VectorStore...")
        vector_store = VectorStore(
            persistence_directory=test_dir,
            collection_name="test_collection",
            embedding_model_id="BAAI/bge-small-en-v1.5",
            search_type="hybrid",
            batch_size=3  # Small batch size to test batching
        )

        # Generate sample documents
        logger.info("Generating sample documents...")
        sample_docs = generate_sample_documents(5)

        # Add documents to the vector store
        logger.info("Adding documents to vector store...")
        vector_store.add_documents(sample_docs)

        # Get collection info
        info = vector_store.get_collection_info()
        logger.info(f"Collection info: {info}")

        # Verify document count
        if info['count'] != 5:
            logger.warning(f"Expected 5 documents, but found {info['count']}")

        # Test querying
        logger.info("Testing query functionality...")
        query_text = "vector database search"
        search_results = vector_store.query(query_text, n_results=3)

        logger.info(f"Query: '{query_text}'")
        logger.info(f"Got {len(search_results)} results")

        # Print the search results
        for i, result in enumerate(search_results):
            logger.info(f"Result {i+1}:")
            logger.info(f"  Text: {result['text']}")
            logger.info(f"  Score: {result['relevance_score']}")
            logger.info(f"  Metadata: {result['metadata']}")

        # Test with filter
        logger.info("Testing query with metadata filter...")
        filter_query = vector_store.query(
            query_text="machine learning",
            n_results=2,
            metadata_filter={"category": "category_1"}
        )

        logger.info(f"Filter query results: {len(filter_query)}")

        # Display filter query results
        for i, result in enumerate(filter_query):
            logger.info(f"Filtered Result {i+1}:")
            logger.info(f"  Text: {result['text']}")
            logger.info(f"  Score: {result['relevance_score']}")
            logger.info(f"  Category: {result['metadata'].get('category')}")

        # Test extreme cases
        logger.info("Testing empty query...")
        empty_results = vector_store.query("", n_results=1)
        logger.info(f"Empty query returned {len(empty_results)} results")

        logger.info("Tests completed successfully!")

    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        raise
    finally:
        # Uncomment this to clean up after testing
        # if os.path.exists(test_dir):
        #     import shutil
        #     shutil.rmtree(test_dir)
        #     logger.info(f"Cleaned up test directory: {test_dir}")
        pass

if __name__ == "__main__":
    logger.info("Starting VectorStore test...")
    test_vector_store()
    logger.info("Test completed.")