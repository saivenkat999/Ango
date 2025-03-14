from typing import Dict, Any, List, Optional, Literal
from agno.agent import Agent, RunResponse
from agno.models.openai import OpenAIChat
from agno.tools import Toolkit
from pydantic import BaseModel, Field
import logging
from src.models.response_settings import ResponseSettings
from src.utils.vector_store import VectorStore

from .document_processor_agent import DocumentProcessorAgent
from .retriever_agent import RetrieverAgent, RetrievalQuery

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OrchestratorAgent")

class QueryAnalysisResult(BaseModel):
    """Structured result from user query analysis."""
    refined_query: str = Field(description="The refined, clear version of the user's query")
    search_strategy: str = Field(description="Strategy for searching the knowledge base")
    meta_filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters to apply")
    is_document_management: bool = Field(default=False, description="Whether the query is about document management")
    expected_result_type: str = Field(description="Type of result expected (e.g., 'factual', 'summary', 'explanation')")

# Define response format types
ResponseFormat = Literal["concise", "balanced", "detailed"]

class ResponseGeneratorToolkit(Toolkit):
    """Toolkit for generating responses based on retrieved information."""
    
    def __init__(self, response_settings: ResponseSettings):
        super().__init__(name="response_generator")
        self.response_settings = response_settings
        self.register(self.generate_response)
    
    def generate_response(self, context: str, query: str) -> Dict[str, Any]:
        """
        Generate a response based on the retrieved context and query.
        
        Args:
            context: The retrieved context information
            query: The user's original query
            
        Returns:
            Formatted response dict
        """
        # Create a prompt for generating a response
        prompt = self._create_response_prompt(query, context, self.response_settings)
        
        # Return a properly structured response object that Agno can use
        return {
            "response": prompt,  # The Agno agent will process this prompt to generate the actual response
            "prompt": prompt,
            "query": query,
            "context": context,
            "settings": self.response_settings.dict()
        }
    
    def _create_response_prompt(self, query: str, context: str, settings: ResponseSettings) -> str:
        """
        Create a prompt for generating a response with the appropriate format.
        
        Args:
            query: The user's query
            context: Retrieved context information
            settings: Response format settings
            
        Returns:
            Formatted prompt for the agent
        """
        # Define formatting instructions based on verbosity
        formatting_instructions = {
            "concise": """
                Write a VERY brief and direct answer. 
                - Use plain language with minimal technical terms
                - Maximum 2-3 sentences total
                - No introductions or background information
                - Just give the direct answer to the query
                - Avoid all unnecessary words and phrases
            """,
            "balanced": """
                Provide a complete but balanced response. Include relevant details while avoiding excessive information.
                Aim for around 200-300 words. Use clear headings and bullet points when appropriate.
                Explain key concepts but avoid unnecessary depth.
            """,
            "detailed": """
                Provide a comprehensive and detailed response. Include in-depth explanations and context.
                Use multiple headings, lists, and well-structured sections to organize information.
                Explore relevant related concepts and provide a thorough analysis.
            """
        }
        
        # Define citation instructions
        citation_instructions = """
            Include specific citations to source documents using the format [Document X: Source Name (page Y)].
            Place citations directly after the information they support.
        """ if settings.include_citations else """
            Do not include explicit citations in your response.
        """
        
        # Define sources section instructions
        sources_section_instructions = """
            Include a "Sources" section at the end listing all documents referenced.
        """ if settings.include_sources_section else ""
        
        # Define length constraint
        length_constraint = f"""
            Keep your response to approximately {settings.max_length} characters in length.
        """ if settings.max_length else ""
        
        # Add strict RAG instructions
        rag_instructions = """
            CRITICAL: Only use information from the provided context. DO NOT include any information that is not present in the retrieved documents.
            If the retrieved documents don't contain enough information to fully answer the query, acknowledge this limitation explicitly.
            Never hallucinate or make up information not present in the retrieved documents.
        """
        
        # Create the prompt
        prompt = f"""
        Generate a {settings.verbosity} response to the user's query based on the retrieved information.
        
        User Query: "{query}"
        
        Retrieved Context:
        {context}
        
        {rag_instructions}
        
        Your response should:
        1. Directly answer the query with relevant information from the retrieved documents
        2. {formatting_instructions[settings.verbosity]}
        3. {citation_instructions}
        4. {sources_section_instructions}
        5. {length_constraint}
        6. If information is contradictory between sources, acknowledge this and explain the different perspectives
        7. If the retrieved documents don't contain an answer to the question, clearly state this and do not make up information
        8. Format your response in a clean, human-readable way with proper spacing and structure
        
        Remember: You are a Retrieval-Augmented Generation system. You can ONLY provide information that exists in the retrieved documents.
        """
        
        return prompt

# Maintain the old class name as an alias for backward compatibility
ResponseGeneratorTool = ResponseGeneratorToolkit

class OrchestratorAgent:
    """
    Agent responsible for orchestrating the document processing and retrieval.
    
    This agent:
    1. Takes user input and determines appropriate actions
    2. Transforms vague queries into detailed instructions
    3. Coordinates document processor and retriever agents
    4. Formats information for the user
    """
    
    def __init__(self, 
                 model_id: str = "gpt-4o",
                 documents_dir: str = "./data/documents",
                 vector_store: VectorStore = None,
                 default_response_format: ResponseFormat = "balanced"):
        """
        Initialize the Orchestrator Agent.
        
        Args:
            model_id: ID of the OpenAI model to use
            documents_dir: Directory containing PDF documents
            vector_store: Vector store instance for document storage and retrieval
            default_response_format: Default verbosity level for responses
        """
        # Set up agent config
        self.model_id = model_id
        self.documents_dir = documents_dir
        self.vector_store = vector_store
        
        # Set default response settings
        self.default_response_settings = ResponseSettings(
            verbosity=default_response_format
        )
        
        # Initialize child agents
        self.document_processor = DocumentProcessorAgent(
            model_id=model_id,
            documents_dir=documents_dir,
            vector_store=self.vector_store
        )
        
        self.retriever = RetrieverAgent(
            model_id=model_id,
            vector_store=self.vector_store
        )
        
        # Create tools for the agent
        tools = self.get_tools(self.default_response_settings)
        
        # Initialize the agent with specialized instructions
        self.agent = Agent(
            name="Orchestrator",
            model=OpenAIChat(id=model_id),
            description="Master orchestrator agent that coordinates document processing and retrieval operations",
            instructions=[
                "You are the primary coordinator of a multi-agent system for document retrieval.",
                
                "Step 1: Query Analysis",
                "- Analyze and interpret the user's query to understand their information needs",
                "- Refine vague or unclear queries into detailed, actionable instructions",
                "- Determine if the query requires document processing or information retrieval",
                "- Identify key concepts, entities, and metadata filters relevant to the query",
                
                "Step 2: Task Delegation",
                "- Route document processing requests to the Document Processor Agent",
                "- Route information retrieval queries to the Retriever Agent", 
                "- Monitor the progress and results of delegated tasks",
                "- Ensure consistent communication format between agents",
                
                "Step 3: Response Generation",
                "- Generate comprehensive and accurate responses based on retrieved information",
                "- Format responses according to specified verbosity level and citation preferences",
                "- Ensure responses directly address the user's query with relevant information",
                "- Apply appropriate formatting for readability and clarity",
                "- Include proper citations and source references when required",
                
                "Step 4: Conversation Management",
                "- Maintain context across multiple interactions in the conversation",
                "- Connect related queries and build upon previous information",
                "- Request clarification when necessary",
                "- Handle system commands and formatting requests appropriately",
                
                "Important Guidelines:",
                "- Always prioritize accuracy and relevance in responses",
                "- Only provide information that can be sourced from the knowledge base",
                "- Acknowledge limitations when requested information is not available",
                "- Present information in a structured, coherent manner",
                "- Format responses according to the specified verbosity level",
                "- Provide appropriate citations for all retrieved information when required"
            ],
            team=[self.document_processor.agent, self.retriever.agent],  # Set up team structure
            tools=tools,  # Pass tools to the agent
            read_chat_history=True,  # Enable chat history access
            structured_outputs=QueryAnalysisResult,  # For internal analysis
            markdown=True,
            debug_mode=True,  # Enable debug mode to see detailed logs
            show_tool_calls=True  # Show when and how tools are called
        )
    
    def process_documents(self) -> str:
        """
        Process documents using the Document Processor Agent.
        
        Returns:
            Status message about document processing
        """
        # Delegate document processing to the document processor agent
        result = self.document_processor.process_documents()
        
        if result.success:
            return f"✅ {result.details}"
        else:
            return f"❌ Document processing failed: {result.error_message}"
    
    def analyze_query(self, query: str) -> QueryAnalysisResult:
        """
        Analyze the user query to determine the appropriate action.
        
        Args:
            query: User's query text
            
        Returns:
            QueryAnalysisResult with analysis of the query
        """
        # Use the agent to analyze the query
        prompt = f"""
        Analyze the following user query and provide a structured response:
        
        User Query: "{query}"
        
        Your task:
        1. Determine if this is a document management request or an information retrieval request
        2. Refine the query to be more specific and detailed if necessary
        3. Suggest an appropriate search strategy and metadata filters if applicable
        4. Identify the expected type of result (factual answer, summary, explanation, etc.)
        """
        
        result = self.agent.run(prompt)
        
        # Convert result to QueryAnalysisResult
        # Check if we're dealing with document management
        is_document_management = any(keyword in query.lower() for keyword in 
                                   ["process documents", "index documents", "add documents", 
                                    "update knowledge base", "rebuild", "reindex"])
        
        # Simple refinement for demonstration
        refined_query = query
        if "?" not in query:
            refined_query += "?"
            
        # Create analysis result
        analysis = QueryAnalysisResult(
            refined_query=refined_query,
            search_strategy="semantic_search",
            meta_filters=None,
            is_document_management=is_document_management,
            expected_result_type="factual" if "what" in query.lower() or "who" in query.lower() else "explanation"
        )
        
        return analysis
    
    def configure_response_settings(self, 
                                   verbosity: Optional[ResponseFormat] = None,
                                   include_citations: Optional[bool] = None,
                                   include_sources_section: Optional[bool] = None,
                                   max_length: Optional[int] = None,
                                   format_markdown: Optional[bool] = None) -> ResponseSettings:
        """
        Configure response settings with custom parameters.
        
        Args:
            verbosity: How detailed the response should be
            include_citations: Whether to include source citations
            include_sources_section: Whether to include a dedicated sources section
            max_length: Maximum length of the response in characters
            format_markdown: Whether to format the response using markdown
            
        Returns:
            Updated response settings
        """
        # Start with default settings
        settings = self.default_response_settings.copy()
        
        # Update with provided parameters
        if verbosity is not None:
            settings.verbosity = verbosity
        if include_citations is not None:
            settings.include_citations = include_citations
        if include_sources_section is not None:
            settings.include_sources_section = include_sources_section
        if max_length is not None:
            settings.max_length = max_length
        if format_markdown is not None:
            settings.format_markdown = format_markdown
            
        return settings
    
    def process_user_query(self, 
                          query: str, 
                          response_format: Optional[ResponseFormat] = None,
                          include_citations: Optional[bool] = None,
                          include_sources_section: Optional[bool] = None,
                          max_length: Optional[int] = None) -> str:
        """
        Process a user query and return an appropriate response.
        
        Args:
            query: User's query text
            response_format: Verbosity level for the response
            include_citations: Whether to include source citations
            include_sources_section: Whether to include a dedicated sources section
            max_length: Maximum length of the response in characters
            
        Returns:
            Formatted response text
        """
        # Handle document processing command - check if this is a document processing request
        processed_query = query.lower().strip()
        if processed_query == "process documents" or (
            "document" in processed_query and 
            any(cmd in processed_query for cmd in ["process", "analyze", "index", "scan", "read", "parse"])
        ):
            return self.process_documents()
        
        # Configure response settings for this query
        response_settings = self.configure_response_settings(
            verbosity=response_format,
            include_citations=include_citations,
            include_sources_section=include_sources_section,
            max_length=max_length
        )
        
        # Log that we're processing a user query
        logger.info(f"Processing user query: {query}")
        
        try:
            # Analyze the query to refine it
            analysis = self.analyze_query(query)
            logger.debug(f"Query analysis complete: {analysis}")
            
            # Define retrieval query
            retrieval_query = RetrievalQuery(
                query=analysis.refined_query,
                n_results=5,
                metadata_filter=analysis.meta_filters
            )
            
            # Retrieve relevant information
            retrieval_result = self.retriever.retrieve(retrieval_query)
            logger.debug(f"Retrieval complete: success={retrieval_result.success}")
            
            # Check if retrieval was successful and contains meaningful information
            if not retrieval_result.success or not retrieval_result.context or retrieval_result.context.startswith("No information"):
                return "I couldn't find any relevant information in the knowledge base to answer your query."
            
            # Create a direct instruction prompt for the RAG task 
            # This is a comprehensive instruction rather than a dynamically constructed prompt
            rag_prompt = f"""
            Based on the following retrieved context from our knowledge base, please answer the user's query:
            
            USER QUERY: {query}
            
            RETRIEVED CONTEXT:
            {retrieval_result.context}
            
            Follow these guidelines:
            1. Only include information that is present in the retrieved context
            2. Format your response according to the '{response_settings.verbosity}' verbosity level
            3. Be direct and concise in your answer
            4. If the information in the context is insufficient, acknowledge this limitation
            """
            
            # Run the agent with this direct prompt
            response = self.agent.run(rag_prompt)
            
            # Access the content using get_content_as_string() which properly handles all response types
            return response.get_content_as_string()
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return f"Error processing your query: {str(e)}"

    @staticmethod
    def get_tools(response_settings: ResponseSettings):
        """Get orchestration tools for the Agno agent."""
        return [ResponseGeneratorToolkit(response_settings)] 