from typing import Dict, Any, Optional, Literal, List
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import Toolkit
from pydantic import BaseModel, Field
import logging
from src.utils.config import Config
from src.utils.vector_store import VectorStore

from .document_processor_agent import DocumentProcessorAgent
from .retriever_agent import RetrieverAgent, RetrievalQuery

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OrchestratorAgent")

class QueryAnalysisResult(BaseModel):
    """Structured result from user query analysis."""
    refined_query: str = Field(description="The refined, clear version of the user's query")
    meta_filters: Optional[Dict[str, Any]] = Field(None, description="Optional metadata filters to apply")
    expected_result_type: str = Field(description="Type of result expected (e.g., 'factual', 'summary', 'explanation')")

# Define response format types
ResponseFormat = Literal["concise", "balanced", "detailed"]

class ResponseSettings(BaseModel):
    """Settings for customizing response format and content."""
    verbosity: ResponseFormat = Field(
        default=Config.RESPONSE_FORMAT, 
        description="How detailed the response should be"
    )
    max_length: Optional[int] = Field(
        default=Config.MAX_RESPONSE_LENGTH, 
        description="Maximum length of the response in characters (approximate)"
    )
    format_markdown: bool = Field(
        default=Config.FORMAT_MARKDOWN, 
        description="Whether to format the response using markdown"
    )
    
    def format_response(self, response) -> str:
        """
        Format a response based on these settings.
        
        Args:
            response: The raw response text or Agno RunResponse object
            
        Returns:
            A formatted, human-readable response
        """
        # If response is None or empty, return a default message
        if response is None:
            return "No response generated."
        
        # Convert response to string, handling various response types
        if not isinstance(response, str):
            try:
                # Simple string conversion, which should work for most objects
                response = str(response)
            except Exception as e:
                logger.error(f"Error converting response to string: {str(e)}")
                return f"Error formatting response: {str(e)}"
        
        # Ensure response is a string at this point
        if not isinstance(response, str):
            return "Error: Could not convert response to string."
        
        # Clean up the response
        response = response.strip()
        
        # For concise responses, ensure they're really concise
        if self.verbosity == "concise":
            # Strip any markdown headers
            response = response.replace("# ", "").replace("## ", "").replace("### ", "")
            
            # Split into sentences and limit if too many
            sentences = [s.strip() for s in response.split('.') if s.strip()]
            if len(sentences) > 5:
                sentences = sentences[:5]
                response = ". ".join(sentences) + "."
            
            # Remove any citation blocks at the end
            if "Sources:" in response:
                response = response.split("Sources:")[0].strip()
        
        # Apply max_length constraint if specified
        if self.max_length and len(response) > self.max_length:
            response = response[:self.max_length-3] + "..."
            
        return response


class ResponseGeneratorToolkit(Toolkit):
    """Toolkit for generating well-structured responses based on retrieved information."""

    def __init__(self, response_settings: ResponseSettings, model: OpenAIChat):
        super().__init__(name="response_generator")
        self.response_settings = response_settings
        self.model = model
        self.register(self.generate_response)

    def generate_response(self, context: str, query: str, expected_result_type: str = "factual") -> Dict[str, Any]:
        """
        Generate a response based on retrieved context information and the original query.

        Args:
            context: The retrieved context information
            query: The user's original query
            expected_result_type: Type of result expected (factual, procedural, etc.)

        Returns:
            Dictionary containing:
                - response: The generated response text
                - query: The original user query
                - success: Whether generation was successful
        """
        try:
            # Create a prompt for generating a response
            prompt = self._create_response_prompt(query, context,
                                                 self.response_settings,
                                                 expected_result_type)

            # Generate the actual response using the model
            if self.model:
                # Instead of trying to use Message directly, call OpenAI API through client
                try:
                    # If we have direct access to the openai client
                    if hasattr(self.model, 'client'):
                        # Use the OpenAI client
                        response = self.model.client.chat.completions.create(
                            model=self.model.id,
                            messages=[{"role": "user", "content": prompt}]
                        )
                        response_text = response.choices[0].message.content
                        logger.debug(f"Got string content from response: {len(response_text)} chars")
                    else:
                        # Use the model's invoke method with simple string conversion handling
                        from agno.models.message import Message
                        message = Message(role="user", content=prompt)
                        response = self.model.invoke([message])
                        
                        # Extract content as string - use string representation if it's not already a string
                        if hasattr(response, 'content'):
                            if isinstance(response.content, str):
                                response_text = response.content
                                logger.debug(f"Got string content from response: {len(response_text)} chars")
                            else:
                                # Convert non-string content to string (e.g., dict to string)
                                logger.warning(f"response.content is not a string but {type(response.content)}. Converting to string.")
                                logger.debug(f"Content value: {response.content}")
                                response_text = str(response.content)
                        elif hasattr(response, 'choices') and len(response.choices) > 0:
                            response_text = response.choices[0].message.content
                            logger.debug(f"Got content from response.choices: {len(response_text)} chars")
                        else:
                            logger.warning("Could not extract content from model response")
                            response_text = "No response generated."
                except Exception as e:
                    logger.error(f"Error with model.invoke: {str(e)}")
                    # Fallback in case of error
                    response_text = f"Error generating response: {str(e)}"
            else:
                logger.warning("No model provided to ResponseGeneratorToolkit")
                response_text = "No response model available to generate a response."

            # Format the response according to settings
            formatted_response = self.response_settings.format_response(response_text)

            return {
                "response": formatted_response,
                "query": query,
                "success": True
            }

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return {
                "response": f"I found relevant information but encountered an error generating a response: {str(e)}",
                "query": query,
                "success": False,
                "error": str(e)
            }

    def _create_response_prompt(self, query: str, context: str,
                               settings: ResponseSettings,
                               expected_result_type: str = "factual") -> str:
        """Create a prompt for generating a response with the appropriate format."""
        # Define formatting instructions based on verbosity
        formatting_instructions = {
            "concise": "Write a VERY brief and direct answer (2-3 sentences). Use plain language with minimal technical terms. No introductions or background information.",
            "balanced": "Provide a complete but balanced response with relevant details (200-300 words). Use clear headings and bullet points when appropriate.",
            "detailed": "Provide a comprehensive and detailed response with in-depth explanations. Use multiple headings, lists, and well-structured sections."
        }

        # Result type specific instructions
        result_type_instructions = {
            "factual": "Focus on accurate facts and specific details. Be precise and direct.",
            "procedural": "Present clear step-by-step instructions in a logical sequence. Number steps when appropriate.",
            "conceptual": "Explain concepts thoroughly with clear definitions and examples.",
            "opinion": "Present a balanced view of different perspectives based only on the retrieved information."
        }

        # Define length constraint
        length_constraint = (
            f"Keep your response to approximately {settings.max_length} characters in length."
            if settings.max_length else
            ""
        )

        # Create the prompt
        prompt = f"""
        Generate a {settings.verbosity} response to the user's query based on the retrieved information.

        User Query: "{query}"

        Retrieved Context:
        {context}

        CRITICAL: Only use information from the provided context. DO NOT include any information not present in the documents.

        Your response should:
        1. Directly answer the query with relevant information from the retrieved documents
        2. {formatting_instructions[settings.verbosity]}
        3. {result_type_instructions.get(expected_result_type, result_type_instructions["factual"])}
        4. {length_constraint}
        5. If information is contradictory between sources, acknowledge this and explain the different perspectives
        6. If the retrieved documents don't contain an answer, clearly state this and do not make up information

        Remember: You are a Retrieval-Augmented Generation system. You can ONLY provide information from the retrieved documents.
        """

        return prompt


class QueryAnalyzerToolkit(Toolkit):
    """
    Toolkit for analyzing user queries to optimize retrieval.

    This toolkit analyzes queries to extract key information that helps
    tailor the retrieval process to the specific query characteristics.
    It determines query refinements, metadata filters, and expected result types
    to improve search relevance and response quality.
    """

    def __init__(self, model: OpenAIChat):
        super().__init__(name="query_analyzer")
        self.register(self.analyze_query)
        self.model = model

    def analyze_query(self, query: str) -> str:
        """
        Analyze a user query to extract key information for optimizing retrieval.

        Args:
            query: The original user query text

        Returns:
            A JSON string containing refined query, metadata filters, and expected result type
        """
        try:
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze the following user query:

            "{query}"

            Your task is to:
            1. Refine the query for optimal retrieval
            2. Identify any metadata filters that should be applied
            3. Identify the expected result type (factual, conceptual, procedural, opinion)
            4. Identify the document type or subject domain the query is about

            Respond with a JSON object containing:
            {{
            "refined_query": "Your refined query here",
            "meta_filters": {{"key1": "value1", "key2": "value2"}},
            "expected_result_type": "factual|conceptual|procedural|opinion",
            "document_type": "The document type or subject domain this query is about"
            }}
            """

            # Get response from model
            response_text = ""
            if self.model:
                try:
                    # Use the model's invoke method
                    from agno.models.message import Message
                    message = Message(role="user", content=analysis_prompt)
                    analysis_response = self.model.invoke([message])
                    logger.info(f"Analysis response: {analysis_response}")
                    
                    # Extract content as string
                    if hasattr(analysis_response, 'choices') and analysis_response.choices:
                        response_text = analysis_response.choices[0].message.content
                        logger.info(f"Got string content from response: {len(response_text)} chars")
                    else:
                        logger.info("Could not extract content from model response")
                        response_text = "No response generated."
                except Exception as e:
                    logger.error(f"Error with model.invoke: {str(e)}")
            
            # Parse JSON from response
            import json
            import re
            
            # Try to find JSON in the response
            json_match = re.search(r'({.*})', response_text.replace('\n', ' '), re.DOTALL)
            if json_match:
                analysis_dict = json.loads(json_match.group(1))
                
                result = {
                    "refined_query": analysis_dict.get("refined_query", query),
                    "meta_filters": analysis_dict.get("meta_filters", {}),
                    "expected_result_type": analysis_dict.get("expected_result_type", "factual"),
                    "document_type": analysis_dict.get("document_type", "unknown")
                }
                
                # Enhance metadata filters based on detected document type
                doc_type = result.get("document_type", "").lower()
                meta_filters = result.get("meta_filters", {})
                
                # If document type is detected, add it to metadata filters
                if doc_type and doc_type != "unknown":
                    meta_filters["document_type"] = doc_type
                
                # Detect specific product or technology
                if "odbc" in query.lower() or "odbc" in doc_type:
                    meta_filters["technology"] = "odbc"
                elif "jdbc" in query.lower() or "jdbc" in doc_type:
                    meta_filters["technology"] = "jdbc"
                elif "redshift" in query.lower() or "redshift" in doc_type:
                    meta_filters["database"] = "redshift"
                
                # Update metadata filters in result
                result["meta_filters"] = meta_filters
                
                return json.dumps(result)

            # If no valid JSON found, use fallback analysis
            logger.warning("No valid JSON found in response, using fallback analysis")
            
            # Basic fallback logic
            meta_filters = {}
            if "encryption" in query.lower():
                meta_filters["feature"] = "encryption"
            if "ssl" in query.lower() or "tls" in query.lower():
                meta_filters["feature"] = "ssl"
            if any(db in query.lower() for db in ["oracle", "sql server", "odbc", "jdbc"]):
                for db_type in ["oracle", "sqlserver", "odbc", "jdbc"]:
                    if db_type in query.lower():
                        meta_filters["driver_type"] = db_type
                        break
            
            # Determine expected result type
            expected_result_type = "factual"  # default
            if any(term in query.lower() for term in ["how to", "steps", "procedure", "configure"]):
                expected_result_type = "procedural"
            elif any(term in query.lower() for term in ["explain", "concept", "understand"]):
                expected_result_type = "conceptual"
            elif any(term in query.lower() for term in ["opinion", "best", "recommend"]):
                expected_result_type = "opinion"

            result = {
                "refined_query": query,
                "meta_filters": meta_filters,
                "expected_result_type": expected_result_type,
                "document_type": "unknown"
            }
            
            # Enhance metadata filters based on detected document type
            doc_type = result.get("document_type", "").lower()
            meta_filters = result.get("meta_filters", {})
            
            # If document type is detected, add it to metadata filters
            if doc_type and doc_type != "unknown":
                meta_filters["document_type"] = doc_type
            
            # Detect specific product or technology
            if "odbc" in query.lower() or "odbc" in doc_type:
                meta_filters["technology"] = "odbc"
            elif "jdbc" in query.lower() or "jdbc" in doc_type:
                meta_filters["technology"] = "jdbc"
            elif "redshift" in query.lower() or "redshift" in doc_type:
                meta_filters["database"] = "redshift"
            
            # Update metadata filters in result
            result["meta_filters"] = meta_filters
            
            return json.dumps(result)

        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}", exc_info=True)
            # Simple error fallback
            return json.dumps({
                "refined_query": query,
                "meta_filters": {},
                "expected_result_type": "factual",
                "document_type": "unknown"
            })

class OrchestratorAgent:
    """
    Master agent responsible for coordinating the entire RAG system workflow.
    
    This agent serves as the central coordinator for the multi-agent RAG system,
    managing the flow of information between different specialized agents and
    ensuring that user queries are properly processed and answered.
    
    Key responsibilities:
    1. Analyzing and refining user queries to determine information needs
    2. Delegating document processing tasks to the DocumentProcessorAgent
    3. Delegating information retrieval tasks to the RetrieverAgent
    4. Generating comprehensive and accurate responses based on retrieved information
    5. Managing conversation context and handling special commands
    6. Coordinating the overall system workflow and agent interactions
    
    The OrchestratorAgent makes high-level decisions about how to process queries,
    while delegating specialized tasks to other agents that have specific expertise.
    """
    
    def __init__(self, 
                 model_id: str = Config.MODEL_ID,
                 documents_dir: str = Config.DOCUMENTS_DIR,
                 vector_store: VectorStore = None,
                 use_reranking: bool = Config.USE_RERANKING,
                 default_response_format: ResponseFormat = Config.RESPONSE_FORMAT):
        """
        Initialize the Orchestrator Agent with configuration and child agents.
        
        Args:
            model_id: ID of the LLM model to use for the orchestrator
            documents_dir: Directory containing documents to be processed
            vector_store: Vector store for document storage and retrieval
            use_reranking: Whether to use reranking to improve search results
            default_response_format: Default verbosity level for responses
        """
        # Set up basic configuration
        self.model_id = model_id
        self.documents_dir = documents_dir
        self.vector_store = vector_store
        self.default_response_settings = ResponseSettings(verbosity=default_response_format)
        self.default_response_settings.max_length = Config.MAX_RESPONSE_LENGTH
        self.default_response_settings.verbosity = Config.RESPONSE_FORMAT
        self.model = OpenAIChat(id=model_id, api_key=Config.OPENAI_API_KEY)
        
        # Initialize child agents
        self.document_processor = DocumentProcessorAgent(
            model_id=model_id,
            documents_dir=documents_dir,
            vector_store=self.vector_store
        )
        
        self.retriever = RetrieverAgent(
            model_id=model_id,
            vector_store=self.vector_store,
            cohere_api_key=Config.COHERE_API_KEY
        )
        
        # Create toolkits
        self.query_analyzer_toolkit = QueryAnalyzerToolkit(model=self.model)
        self.response_generator_toolkit = ResponseGeneratorToolkit(self.default_response_settings, model=self.model)
        
        # Initialize conversation context tracking
        self.conversation_context = {
            "current_document_focus": None,  # Currently focused document
            "recent_metadata_filters": [],   # Recent metadata filters used
            "recent_topics": [],             # Recent topics discussed
            "query_history": []              # History of previous queries
        }
        
        # Initialize the agent with specialized instructions
        self.agent = Agent(
            name="Orchestrator",
            model=self.model,
            description="Master orchestrator agent that coordinates document processing and retrieval operations",
            instructions=[
                "You are the primary coordinator of a multi-agent system for document retrieval.",
                
                "Step 1: Query Analysis",
                "- Analyze and interpret the user's query to understand their information needs",
                "- Refine vague or unclear queries into detailed, actionable instructions",
                "- Determine if the query requires document processing or information retrieval",
                "- Identify key concepts, entities, and metadata filters relevant to the query",
                "- Determine the expected result type (factual, conceptual, procedural, opinion)",
                "- Provide context about your analysis to the tools to aid their reasoning",
                
                "For DataDirect documentation queries:",
                "- Recognize product names (DataDirect Connect, ODBC, JDBC, ADO.NET)",
                "- Identify specific drivers (Oracle, SQL Server, DB2, etc.)",
                "- Understand common features (connection pooling, SSL/TLS, encryption)",
                "- Categorize query types (configuration, troubleshooting, reference)",
                "- Extract version information when present",
                
                "Step 2: Task Delegation",
                "- Route document processing requests to the Document Processor Agent",
                "- Delegate information retrieval tasks to the RetrieverAgent with appropriate context", 
                "- Provide the RetrieverAgent with your query analysis to inform its search strategy",
                "- Allow the RetrieverAgent to make decisions about search methods and result evaluation",
                "- Monitor the progress and results of delegated tasks",
                
                "Step 3: Response Generation",
                "- Generate comprehensive and accurate responses based on retrieved information",
                "- Format responses according to specified verbosity level",
                "- Ensure responses directly address the user's query with relevant information",
                "- Apply appropriate formatting for readability and clarity",
                "- Use only information from the retrieved documents in your responses",
                
                "For DataDirect documentation responses:",
                "- Include relevant connection string attributes when discussing configuration",
                "- Provide complete code examples when showing implementation steps",
                "- Format error codes and messages in a clear, readable way",
                "- Organize troubleshooting steps in a logical sequence",
                "- Include version compatibility information when relevant",
                
                "Step 4: Conversation Management",
                "- Maintain context across multiple interactions in the conversation",
                "- Connect related queries and build upon previous information",
                "- Request clarification when necessary",
                "- Handle system commands and formatting requests appropriately",
                
                "Important Guidelines:",
                "- Respect the expertise of specialized tools - let them make decisions",
                "- Provide clear context when delegating tasks to the tools",
                "- Always prioritize accuracy and relevance in responses",
                "- Only provide information that can be sourced from the knowledge base",
                "- Acknowledge limitations when requested information is not available",
                "- Present information in a structured, coherent manner"
            ],
            tools=[self.query_analyzer_toolkit, self.response_generator_toolkit],  # Pass tools to the agent
            read_chat_history=True,  # Enable chat history access
            response_model=QueryAnalysisResult,  # For internal analysis
            structured_outputs=True,  # For internal analysis
            markdown=True,
            debug_mode=True,  # Enable debug mode to see detailed logs
            show_tool_calls=True  # Show when and how tools are called
        )
    
    def process_documents(self) -> str:
        """
        Process documents in the specified directory and store them in the vector database.
        
        This method delegates document processing to the DocumentProcessorAgent, which:
        1. Reads documents from the configured documents directory
        2. Chunks the documents into manageable pieces
        3. Processes and embeds the chunks
        4. Stores the embedded chunks in the vector database
        
        Returns:
            A formatted string with the processing results, including:
            - Success/failure status
            - Number of documents processed
            - Number of chunks created
            - Additional details about the processing operation
        """
        logger.info("Orchestrator requesting document processing")
        processing_result = self.document_processor.process_and_store_document(self.documents_dir)
        
        if processing_result.success:
            return f"✅ Successfully processed {processing_result.document_count} documents with {processing_result.chunk_count} chunks.\n\n{processing_result.details}"
        else:
            return f"❌ Document processing failed: {processing_result.error_message}\n\n{processing_result.details if processing_result.details else ''}"
    
    
    def process_query(self, query: str, response_format: Optional[ResponseFormat] = None) -> str:
        """Process a user query through the complete RAG pipeline and return a response."""
        try:
            # Handle special "process documents" command
            if query.lower() == "process documents":
                return self.process_documents()

            # Track query in history
            self.conversation_context["query_history"].append(query)
            if len(self.conversation_context["query_history"]) > 10:
                # Keep only the last 10 queries
                self.conversation_context["query_history"] = self.conversation_context["query_history"][-10:]

            # Set response settings based on provided format or default
            response_settings = self.default_response_settings
            if response_format:
                response_settings.verbosity = response_format

            logger.info(f"Processing query: '{query}'")

            # Step 1: Analyze the query using query_analyzer_toolkit
            logger.info(f"Analyzing query: '{query}'")
            
            # Enhance the query analysis with conversation context
            enhanced_query = query
            # If there's a focus on a specific document, add context
            if self.conversation_context.get("current_document_focus"):
                doc_focus = self.conversation_context["current_document_focus"]
                # Only add context if it's not already in the query
                if doc_focus.lower() not in query.lower():
                    enhanced_query = f"{query} [Context: Referring to {doc_focus}]"
                    logger.info(f"Enhanced query with document context: '{enhanced_query}'")
            
            analysis_result = self.query_analyzer_toolkit.analyze_query(enhanced_query)
            
            try:
                import json
                analysis_dict = json.loads(analysis_result)
                refined_query = analysis_dict.get("refined_query", query)
                meta_filters = analysis_dict.get("meta_filters", {})
                expected_result_type = analysis_dict.get("expected_result_type", "factual")
                document_type = analysis_dict.get("document_type", "unknown")
                
                # Update conversation context with detected document type
                if document_type and document_type != "unknown":
                    self.conversation_context["current_document_focus"] = document_type
                    logger.info(f"Updated document focus to: {document_type}")
                
                # Store metadata filters in context for future reference
                if meta_filters:
                    self.conversation_context["recent_metadata_filters"].append(meta_filters)
                    # Keep only the last 5 filters
                    if len(self.conversation_context["recent_metadata_filters"]) > 5:
                        self.conversation_context["recent_metadata_filters"] = self.conversation_context["recent_metadata_filters"][-5:]
                
                # Check for follow-up questions
                is_followup = self._is_followup_question(query)
                if is_followup and self.conversation_context["recent_metadata_filters"]:
                    # Enhance metadata filters with previous context
                    prev_filters = self.conversation_context["recent_metadata_filters"][-1]
                    logger.info(f"Detected follow-up question, enhancing with previous filters: {prev_filters}")
                    # Only add previous filters if they don't conflict with current ones
                    for key, value in prev_filters.items():
                        if key not in meta_filters:
                            meta_filters[key] = value
                
                logger.info(f"Query analysis complete. Refined query: '{refined_query}'")
                logger.info(f"Metadata filters: {meta_filters}")
                logger.info(f"Expected result type: {expected_result_type}")
                logger.info(f"Document type: {document_type}")
            except Exception as e:
                logger.error(f"Error parsing analysis result: {str(e)}", exc_info=True)
                # Fall back to original query
                refined_query = query
                meta_filters = {}
                expected_result_type = "factual"
                document_type = "unknown"
            
            # Step 2: Retrieve information using the RetrieverAgent
            logger.info(f"Retrieving information for refined query: '{refined_query}'")
            
            # Create the RetrievalQuery object with only the required fields
            retrieval_query = RetrievalQuery(
                query=refined_query,
                n_results=Config.MAX_RETRIEVAL_RESULTS,
                metadata_filter=meta_filters,
                search_type="hybrid",
                expected_result_type=expected_result_type
                #query_context=f"Original query: {query}"
            )
            
            # Use the RetrieverAgent to retrieve information
            retrieval_result = self.retriever.retrieve(retrieval_query)
            
            if not retrieval_result.success:
                logger.warning(f"Retrieval failed: {retrieval_result.error_message}")
                error_message = retrieval_result.error_message or "No information found"
                if "string indices must be integers" in error_message:
                    # This is a known error indicating a format mismatch
                    error_message = "An internal format error occurred. Please try again."
                
                # Try a direct fallback search with simpler parameters
                try:
                    logger.info("Attempting fallback direct search")
                    # Use the retriever's direct_query method which bypasses agent logic
                    direct_results = self.retriever.direct_query(
                        query_text=refined_query,
                        n_results=Config.MAX_RETRIEVAL_RESULTS,
                        metadata_filter=meta_filters,
                        search_type="vector"  # Fall back to more reliable vector search
                    )
                    
                    if direct_results and len(direct_results) > 0:
                        # We got some results, format them into context
                        logger.info(f"Fallback search successful, found {len(direct_results)} results")
                        context = self._format_direct_results(direct_results, expected_result_type)
                        
                        # Generate response from context
                        response_result = self.response_generator_toolkit.generate_response(
                            context=context,
                            query=query,
                            expected_result_type=expected_result_type
                        )
                        
                        if response_result.get("success", False):
                            formatted_response = response_settings.format_response(response_result["response"])
                            return formatted_response
                    
                    # If we reach here, the fallback didn't work either
                    logger.warning("Fallback search returned no results")
                except Exception as fallback_error:
                    logger.error(f"Fallback search failed: {str(fallback_error)}", exc_info=True)
                
                # If all fails, return the error
                return f"I couldn't find information related to your query. {error_message}"
                
            if not retrieval_result.context or retrieval_result.context.strip() == "":
                logger.warning("Retrieval returned empty context")
                return "I couldn't find specific information related to your query in our knowledge base."
            
            # Step 3: Generate a response using the response_generator_toolkit
            logger.info("Generating response from retrieved information")
            response_result = self.response_generator_toolkit.generate_response(
                context=retrieval_result.context,
                query=query,
                expected_result_type=expected_result_type
            )
            
            if not response_result.get("success", False):
                logger.warning(f"Response generation failed: {response_result.get('error', 'Unknown error')}")
                # Fallback to returning the context directly
                formatted_response = response_settings.format_response(
                    f"Here's what I found about your query:\n\n{retrieval_result.context}"
                )
                return formatted_response
            
            formatted_response = response_settings.format_response(response_result["response"])
            return formatted_response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return f"I encountered an error while processing your query: {str(e)}"

    def _format_direct_results(self, results: List[Dict[str, Any]], expected_result_type: str = "factual") -> str:
        """Format direct results from vector store into context for response generation."""
        sections = []
        
        for i, result in enumerate(results):
            # Format the text content
            content = result.get("text", "")
            if not content:
                continue
                
            # Format based on expected_result_type
            if expected_result_type == "conceptual":
                section = f"**Information {i+1}**\n\n{content}\n\n"
            elif expected_result_type == "procedural":
                section = f"**Procedure {i+1}**\n\n{content}\n\n"
            elif expected_result_type == "opinion":
                section = f"**Perspective {i+1}**\n\n{content}\n\n"
            else:
                section = f"**Document {i+1}**\n\n{content}\n\n"
            
            sections.append(section)
            
        if not sections:
            return "No relevant information found."
            
        return "\n".join(sections)

    def _get_focus_instruction(self, result_type: str) -> str:
        """Helper to get appropriate focus instruction based on result type"""
        if result_type == "factual":
            return "accurate facts and specific details"
        elif result_type == "procedural":
            return "clear step-by-step instructions"
        elif result_type == "conceptual":
            return "thorough explanations of concepts"
        elif result_type == "opinion":
            return "balanced presentation of perspectives"
        else:
            return "relevant information"
    
    @staticmethod
    def get_tools(response_settings: ResponseSettings):
        """Get tools for the Agno agent."""
        return [QueryAnalyzerToolkit(), ResponseGeneratorToolkit(response_settings)]

    def _is_followup_question(self, query: str) -> bool:
        """
        Determine if the query is a follow-up question based on linguistic patterns.
        
        Args:
            query: The user query text
            
        Returns:
            True if the query appears to be a follow-up question, False otherwise
        """
        # Follow-up patterns
        followup_patterns = [
            r'^(what|how|why|when|where|who|which) (about|if|is|are|does|do|can|could|would|will|has|have)',  # What about, How is, etc.
            r'^(and|also|additionally|besides|furthermore|moreover)',  # Starting with conjunctions
            r'^(can|could|would|will) (you|it|they|we)',  # Can you, Would it, etc.
            r'\b(it|this|that|these|those|they)\b',  # Pronouns without clear referents
            r'^(tell me more|explain further|elaborate|continue|go on)',  # Explicit continuation requests
        ]
        
        query_lower = query.lower().strip()
        
        import re
        # Check for follow-up patterns
        for pattern in followup_patterns:
            if re.search(pattern, query_lower):
                return True
                
        # Check if query is very short (likely a follow-up)
        if len(query_lower.split()) <= 3:
            return True
            
        # Check if query has no interrogative word or verb (incomplete question)
        has_question_element = any(word in query_lower for word in 
                                 ['what', 'how', 'why', 'when', 'where', 'who', 'which', 
                                  'show', 'tell', 'explain', 'describe'])
        if not has_question_element:
            return True
            
        return False

# For backward compatibility - kept for API compatibility
process_user_query = OrchestratorAgent.process_query 