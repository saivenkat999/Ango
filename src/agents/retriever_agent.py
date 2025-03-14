import logging
from typing import List, Dict, Any, Optional, Union
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import Toolkit
from pydantic import BaseModel, Field
import re

from ..utils.vector_store import VectorStore

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RetrieverAgent")

class RetrievalQuery(BaseModel):
    """Structured query for retrieving information from the knowledge base."""
    query: str = Field(description="The query text to search for")
    n_results: int = Field(default=5, description="Number of results to retrieve")
    metadata_filter: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filter")

class RetrievalResult(BaseModel):
    """Structured result from knowledge base retrieval operations."""
    success: bool = Field(description="Whether the retrieval was successful")
    query: str = Field(description="The original query")
    results: List[Dict[str, Any]] = Field(description="The retrieved results")
    context: str = Field(description="Formatted context from the results")
    error_message: Optional[str] = Field(None, description="Error message if retrieval failed")

class InfoRetrievalToolkit(Toolkit):
    """Toolkit for retrieving information from the knowledge base."""
    
    def __init__(self, vector_store: VectorStore):
        super().__init__(name="information_retrieval")
        self.vector_store = vector_store
        self.register(self.retrieve_information)
    
    def retrieve_information(self, query: str, n_results: int = 5, filters: Optional[Dict] = None) -> str:
        """
        Retrieve relevant information from the vector store based on the query.
        
        Args:
            query: The user's query
            n_results: Maximum number of results to return
            filters: Optional metadata filters to apply
            
        Returns:
            A string containing formatted retrieved information
        """
        try:
            # Log the retrieval request
            logger.info(f"Retrieving information for query: '{query}', n_results={n_results}, filters={filters}")
            
            # Request more results to compensate for relevance filtering
            vector_results = self.vector_store.query(
                query_text=query,
                n_results=n_results,
                metadata_filter=filters
            )
            
            # Log the raw results received
            logger.debug(f"Raw vector store results: {vector_results.keys()}")
            
            # Check if we have any results
            if not vector_results["ids"][0] or len(vector_results["ids"][0]) == 0:
                logger.warning(f"No results found for query: {query}")
                return "No information related to your query was found in the knowledge base."
            
            # Structure the results
            structured_results = []
            
            for i in range(len(vector_results["ids"][0])):
                # Skip if index is out of bounds
                if i >= len(vector_results["documents"][0]) or i >= len(vector_results["metadatas"][0]):
                    continue
                    
                # Get document ID, text, and metadata
                doc_id = vector_results["ids"][0][i]
                text = vector_results["documents"][0][i]
                metadata = vector_results["metadatas"][0][i]
                
                # Get distance if available
                distance = None
                if "distances" in vector_results and vector_results["distances"] and len(vector_results["distances"][0]) > i:
                    distance = vector_results["distances"][0][i]
                
                # Calculate relevance score
                relevance_score = 1.0 - distance if distance is not None else None
                
                # Add to structured results
                structured_results.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata,
                    "relevance_score": relevance_score
                })
            
            # Limit to requested number of results
            structured_results = structured_results[:n_results]
            
            # Format the context
            formatted_context = self._format_context(structured_results)
            
            # Log success
            logger.info(f"Successfully retrieved {len(structured_results)} results for query: '{query}'")
            
            # Return the formatted context as a string
            return formatted_context
            
        except Exception as e:
            logger.error(f"Error retrieving information: {str(e)}", exc_info=True)
            error_message = f"Error retrieving information: {str(e)}"
            return error_message
    
    def _format_context(self, results: List[Dict]) -> str:
        """Format the retrieved results into a readable context string."""
        if not results:
            return "No relevant information about this query was found in the knowledge base."
        
        # Create a header section
        header = "### Retrieved Information\n\n"
        
        # Format each result
        sections = []
        
        for i, result in enumerate(results):
            # Format metadata
            source = result["metadata"].get("source", "Unknown")
            file_path = result["metadata"].get("file_path", "Unknown")
            page = result["metadata"].get("page", "Unknown")
            
            # Format the text content
            content = result["text"].strip()
            
            # Remove extra whitespace and newlines
            content = " ".join(content.split())
            
            # Format the section
            section = f"**Document {i+1}**: {source}"
            if page != "Unknown":
                section += f" (page {page})"
            
            # Add relevance score if available
            if result.get("relevance_score") is not None:
                section += f" [Relevance: {result['relevance_score']:.2f}]"
            
            section += f"\n\n{content}\n\n"
            sections.append(section)
        
        # Combine everything
        formatted_context = header + "\n".join(sections)
        
        return formatted_context

class RetrieverAgent:
    """
    Agent responsible for retrieving information from the knowledge base.
    
    This agent:
    1. Takes query instructions from the orchestrator agent
    2. Searches the ChromaDB vector database for relevant content
    3. Returns contextually appropriate results
    """
    
    def __init__(self, 
                 model_id: str = "gpt-4o",
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
        
        # Create tools for the agent
        tools = self.get_tools(self.vector_store)
        
        # Initialize the agent with specialized instructions
        self.agent = Agent(
            name="Retriever Agent",
            model=OpenAIChat(id=model_id),
            description="Specialized agent for retrieving relevant information from the knowledge base",
            instructions=[
                "You are an expert information retrieval specialist responsible for finding the most relevant information from a knowledge base.",
                
                "Step 1: Query Analysis",
                "- Analyze the user's query to understand information needs",
                "- Identify key concepts and entities in the query",
                "- Determine the appropriate search strategy based on query type",
                
                "Step 2: Information Retrieval",
                "- Search the vector database for information relevant to the query",
                "- Retrieve the most semantically similar document chunks",
                "- Filter results based on relevance score to remove irrelevant information",
                "- Ensure a diverse set of sources when appropriate",
                
                "Step 3: Result Organization",
                "- Structure retrieved information in a readable format",
                "- Order results by relevance to the query",
                "- Include metadata like source documents and page numbers",
                "- Maintain context between related document chunks",
                
                "Important Guidelines:",
                "- Prioritize relevance and accuracy above all else",
                "- Retrieve sufficient context to provide complete answers",
                "- Include metadata so information sources can be properly cited",
                "- Report when relevant information cannot be found",
                "- Never fabricate or modify the content of retrieved documents",
                
                "When reporting results:",
                "1. Structure the information in a readable format",
                "2. Provide context for how each result relates to the query",
                "3. Include relevance scores when available",
                "4. Indicate the source document and location for each result"
            ],
            tools=tools,
            structured_outputs=RetrievalResult,
            markdown=True,
            debug_mode=True,
            show_tool_calls=True
        )
    
    def retrieve(self, retrieval_query: RetrievalQuery) -> RetrievalResult:
        """
        Retrieve relevant information from the knowledge base.
        
        Args:
            retrieval_query: Structured query for retrieval
            
        Returns:
            RetrievalResult containing retrieved information
        """
        try:
            # Get or create the tool for retrieving information
            info_tool = InfoRetrievalToolkit(self.vector_store)
            
            # Log retrieval attempt
            logger.info(f"Agent retrieving information for: '{retrieval_query.query}'")
            
            try:
                # Call the tool function directly to get the formatted context
                context = info_tool.retrieve_information(
                    query=retrieval_query.query,
                    n_results=retrieval_query.n_results,
                    filters=retrieval_query.metadata_filter
                )
                
                # Check if the context indicates a failure (error message)
                if context.startswith("Error") or context.startswith("No information"):
                    # Return a failure result if context indicates an error
                    return RetrievalResult(
                        success=False,
                        query=retrieval_query.query,
                        results=[],
                        context=context,
                        error_message=context if context.startswith("Error") else None
                    )
                
                # For a successful retrieval, return a success result
                return RetrievalResult(
                    success=True,
                    query=retrieval_query.query,
                    results=[],  # We won't use the individual results
                    context=context
                )
                
            except ValueError as ve:
                # Handle embedding-specific errors
                error_msg = f"Embedding error: {str(ve)}"
                logger.error(error_msg)
                return RetrievalResult(
                    success=False,
                    query=retrieval_query.query,
                    results=[],
                    context=f"Error generating embeddings for your query. Please try a different query.",
                    error_message=error_msg
                )
            except Exception as e:
                # Handle other query errors
                error_msg = f"Error in retrieval tool: {str(e)}"
                logger.error(error_msg)
                return RetrievalResult(
                    success=False,
                    query=retrieval_query.query,
                    results=[],
                    context=f"Error retrieving information: {str(e)}",
                    error_message=error_msg
                )
            
        except Exception as e:
            # Return error result for any other exceptions
            error_msg = f"Unexpected error in retrieve method: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return RetrievalResult(
                success=False,
                query=retrieval_query.query,
                results=[],
                context="An unexpected error occurred while retrieving information.",
                error_message=error_msg
            )
    
    def _format_context(self, results: List[Dict[str, Any]]) -> str:
        """
        Format the retrieved results into a readable context string.
        
        Args:
            results: List of retrieval results
            
        Returns:
            Formatted context string
        """
        context = ""
        
        if not results:
            return "No information found in the knowledge base related to this query."
        
        # Sort results by relevance score if available
        if results[0].get('distance') is not None:
            sorted_results = sorted(
                results, 
                key=lambda x: float(1.0 - x.get('distance', 0)) if x.get('distance') is not None else 0, 
                reverse=True
            )
            
            # Use a relaxed threshold since we have less control over relevance filtering
            relevance_threshold = 0.5  # More permissive threshold
            filtered_results = [r for r in sorted_results if r.get('distance') is None or (1.0 - r.get('distance', 0)) > relevance_threshold]
            
            # If all results have been filtered out, keep at least the top result
            if not filtered_results and sorted_results:
                filtered_results = [sorted_results[0]]
        else:
            # If we don't have distances, just use the results as they are
            filtered_results = results
        
        # Add a header section
        context += "# Retrieved Information\n\n"
        
        # Format each result
        for i, result in enumerate(filtered_results):
            # Extract metadata
            metadata = result.get('metadata', {})
            source = metadata.get('source', 'Unknown source')
            file_path = metadata.get('file_path', '')
            page_number = metadata.get('page_number', '')
            
            # Format document citation
            doc_citation = f"Document {i+1}"
            if source != 'Unknown source':
                doc_citation += f": {source}"
            if page_number:
                doc_citation += f" (page {page_number})"
            
            # Add relevance score if available
            relevance = ""
            if result.get('distance') is not None:
                relevance_score = 1.0 - result.get('distance', 0)
                if relevance_score > 0:
                    relevance = f" | Relevance: {relevance_score:.2f}"
            
            # Add document separator with title
            context += f"\n## {doc_citation}{relevance}\n\n"
            
            # Add text content with proper formatting
            content = result.get('text', '').strip()
            
            # Clean up content - remove excessive whitespace and newlines
            content = re.sub(r'\s+', ' ', content)
            
            # Add the text with proper markdown formatting
            context += f"{content}\n\n"
            
            # Add source reference
            if file_path:
                context += f"Source: `{file_path}`\n\n"
            
            # Add a separator between documents
            if i < len(filtered_results) - 1:
                context += "---\n\n"
        
        return context
    
    # NOTE: This method is currently not used in the application flow.
    # It's a convenience wrapper around the retrieve method for direct programmatic use.
    def process_query(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Process a query and return relevant information.
        
        Args:
            query: Query text
            n_results: Number of results to retrieve
            
        Returns:
            Dictionary with retrieved information
        """
        # Create structured query
        retrieval_query = RetrievalQuery(
            query=query,
            n_results=n_results
        )
        
        # Perform retrieval
        result = self.retrieve(retrieval_query)
        
        # Return dictionary representation
        return result.dict()

    @staticmethod
    def get_tools(vector_store: VectorStore):
        """Get retrieval tools for the Agno agent."""
        return [InfoRetrievalToolkit(vector_store)]

# For backward compatibility
InfoRetrievalTool = InfoRetrievalToolkit 