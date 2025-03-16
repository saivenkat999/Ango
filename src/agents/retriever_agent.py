import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import Toolkit
from pydantic import BaseModel, Field
from agno.document import Document as AgnoDocument
from ..utils.config import Config

from ..utils.vector_store import VectorStore

# Import proper reranking functionality
from agno.reranker.cohere import CohereReranker

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

    def __init__(self, vector_store: VectorStore, cohere_api_key: str = None):
        super().__init__(name="information_retrieval")
        self.vector_store = vector_store

        # Initialize the CohereReranker with API key if provided
        self.reranker = CohereReranker(
            top_n=None,  # We'll handle top_n manually
            api_key=cohere_api_key,
            model="rerank-multilingual-v3.0"  # Using the latest model
        )

        # Register the retrieve_information function
        self.register(self.retrieve_information)
        
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
            # Log the retrieval request
            logger.info(f"Retrieving information for query: '{retrieval_query.query}', n_results={retrieval_query.n_results}, filters={retrieval_query.metadata_filter}")

            # The VectorStore returns a list of document dictionaries
            document_results = self.vector_store.query(
                query_text=retrieval_query.query,
                n_results=retrieval_query.n_results,
                metadata_filter=retrieval_query.metadata_filter,
                search_type=retrieval_query.search_type,
            )

            # Check if we have any results
            if not document_results or len(document_results) == 0:
                logger.warning(f"No results found for query: {retrieval_query.query}")
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
                logger.warning(f"No valid results remained after validation for query: {retrieval_query.query}")
                return "No valid information related to your query was found in the knowledge base.", []
                
            document_results = validated_results
            logger.debug(f"Validated {len(document_results)} document results")
            
            # Convert dictionary results to Agno documents for processing
            agno_documents = []
            for doc_dict in document_results:
                try:
                    agno_doc = AgnoDocument(
                        content=doc_dict['text'],
                        meta_data=doc_dict['metadata'],
                        name=doc_dict.get('id', f"doc_{len(agno_documents)}")
                    )
                    agno_documents.append(agno_doc)
                except Exception as e:
                    logger.warning(f"Error converting document to AgnoDocument: {str(e)}")
                    continue

            try:
                # Use CohereReranker to rerank documents - this works directly with Agno Documents
                reranked_documents = self.reranker.rerank(query=retrieval_query.query, documents=agno_documents)

                # Check if we received valid results from reranking
                if reranked_documents and len(reranked_documents) > 0:
                    # Limit to requested number of results
                    reranked_documents = reranked_documents[:retrieval_query.n_results]
                    logger.info(f"Successfully reranked documents, highest score: {reranked_documents[0].reranking_score if reranked_documents else 'N/A'}")
                    # Use these for formatted results
                    final_documents = reranked_documents
                else:
                    logger.warning("Reranker returned no documents, falling back to original results")
                    final_documents = agno_documents[:retrieval_query.n_results]

            except Exception as e:
                logger.error(f"Error during reranking: {str(e)}, proceeding with hybrid search results", exc_info=True)
                # If reranking fails, still return the vector search results
                final_documents = agno_documents[:retrieval_query.n_results]

            # Convert Agno Documents to format expected by _format_context
            structured_results = []
            for doc in final_documents:
                result = {
                    "id": doc.name or doc.id if hasattr(doc, 'id') else "unknown",
                    "text": doc.content,
                    "metadata": doc.meta_data
                }
                structured_results.append(result)

            # Format the context
            formatted_context = self._format_context(structured_results, retrieval_query.expected_result_type)

            # Log success
            logger.info(f"Successfully retrieved {len(structured_results)} results for query: '{retrieval_query.query}'")

            # Return both the formatted context and the document results
            return formatted_context, structured_results

        except Exception as e:
            logger.error(f"Error retrieving information: {str(e)}", exc_info=True)
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

        for i, result in enumerate(results):
            # Format the text content
            content = result["text"]

            # Format based on expected_result_type, aligned with the query model types
            if expected_result_type == "conceptual":
                # For conceptual/explanatory content
                section = f"**Information {i+1}**\n\n{content}\n\n"

            elif expected_result_type == "procedural":
                # For procedural content with steps
                section = f"**Procedure {i+1}**\n\n{content}\n\n"

            elif expected_result_type == "opinion":
                # For opinion-based content
                section = f"**Perspective {i+1}**\n\n{content}\n\n"

            else:
                # Default factual format
                section = f"**Document {i+1}**\n\n{content}\n\n"

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
                 vector_store: VectorStore = None,
                 cohere_api_key: Optional[str] = None):
        """
        Initialize the Retriever Agent.

        Args:
            model_id: ID of the OpenAI model to use
            vector_store: Vector store for querying documents
            cohere_api_key: API key for Cohere reranking (optional)
        """
        # Set up agent config
        self.model_id = model_id

        # Initialize vector store connection
        self.vector_store = vector_store

        # Create abilities for the agent
        self.info_retrieval_toolkit = InfoRetrievalToolkit(
            vector_store=vector_store,
            cohere_api_key=cohere_api_key
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