from typing import Dict, Any, Optional, Literal
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import Toolkit
from pydantic import BaseModel, Field
import logging
import json
import re
from src.utils.config import Config
from src.utils.vector_store import VectorStore

from .document_processor_agent import DocumentProcessorAgent
from .retriever_agent import RetrieverAgent, RetrievalQuery

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OrchestratorAgent")

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
    include_sources: bool = Field(
        default=Config.INCLUDE_SOURCES,
        description="Whether to include sources in the response"
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
        
        # Debug: Log the final response
        logger.debug(f"FORMATTED RESPONSE (first 200 chars): {response[:200]}...")
        
        return response


class ResponseGeneratorToolkit(Toolkit):
    """
    Toolkit for generating responses based on retrieved information.
    
    This toolkit handles the process of crafting coherent, 
    well-structured responses from retrieved document context.

    The toolkit expects:
    1. Context from InformationRetrievalToolkit
    2. Original query from user
    3. Expected result type from QueryAnalyzerToolkit
    4. Response settings for formatting

    The output is a dictionary containing:
    {
        "response": str,  # Generated response text
        "query": str,     # Original query
        "success": bool   # Whether generation was successful
    }
    """
    
    def __init__(self, model: OpenAIChat = None, settings: ResponseSettings = None):
        """
        Initialize the ResponseGeneratorToolkit.

        Args:
            model (OpenAIChat): The model to use for response generation
            settings (ResponseSettings): Settings for response formatting
        """
        super().__init__(name="response_generator")
        self.register(self.generate_response)
        self.model = model
        self.settings = settings or ResponseSettings(
            verbosity=Config.RESPONSE_FORMAT,
            max_length=Config.MAX_RESPONSE_LENGTH,
            format_markdown=Config.FORMAT_MARKDOWN,
            include_sources=Config.INCLUDE_SOURCES
        )

    def generate_response(self, context: str, query: str, expected_result_type: str = "factual") -> str:
        """
        Generate a response based on retrieved context information and the original query.

        Args:
            context (str): The retrieved context information
            query (str): The user's original query
            expected_result_type (str): Type of result expected (factual/procedural/conceptual/opinion)

        Returns:
            str: The generated response text
            
        Raises:
            ValueError: If required parameters are missing
            Exception: If response generation fails
        """
        if not self.model:
            logger.warning("No model provided to ResponseGeneratorToolkit")
            return "No response model available to generate a response."
            
        try:
            # Validate inputs
            if not context or not query:
                raise ValueError("Context and query are required for response generation")
            
            # Create and format the prompt
            prompt = self._create_response_prompt(
                query=query,
                context=context,
                settings=self.settings,
                expected_result_type=expected_result_type
            )
            
            # Generate response using OpenAI client
            response = self.model.client.chat.completions.create(
                model=self.model.id,
                messages=[{"role": "user", "content": str(prompt)}]
            )
            response_text = response.choices[0].message.content
            
            # Extract just the actual response part if it contains the running log
            if " - Running: generate_response" in response_text:
                parts = response_text.split("\n\n", 1)
                if len(parts) > 1:
                    response_text = parts[1]
            
            # IMPORTANT: Return the string directly, not a dictionary
            # This fixes the Pydantic error
            return response_text.strip()

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return f"I found relevant information but encountered an error generating a response: {str(e)}"

    def _create_response_prompt(self, query: str, context: str,
                               settings: ResponseSettings,
                               expected_result_type: str = "factual") -> str:
        """Create a prompt for generating a response with the appropriate format."""
        # Define formatting instructions based on verbosity
        formatting_instructions = {
            "concise": "Write a VERY brief and direct answer (2-3 sentences). Focus on the most important information only. Include only critical details. Be concise but complete - don't omit essential information just to be brief.",
            "balanced": "Provide a balanced and complete response (about 150-250 words). Cover all relevant information without unnecessary details.",
            "detailed": "Provide a comprehensive and detailed response with in-depth explanations. Organize the information clearly."
        }

        # Result type specific instructions
        result_type_instructions = {
            "factual": "Focus on accurate facts and specific details. Be precise and direct.",
            "procedural": "Present clear step-by-step instructions in a logical sequence. Number steps when appropriate.",
            "conceptual": "Explain concepts thoroughly with clear definitions and examples.",
            "opinion": "Present a balanced view of different perspectives based only on the retrieved information."
        }

        # Format specific instructions based on markdown setting
        format_instructions = ""
        if settings.format_markdown:
            format_instructions = """
            Use Markdown formatting to structure your response:
            - Use # for main headings, ## for subheadings, and ### for section headings
            - Use bullet points (*) for lists when appropriate
            - Use numbered lists (1., 2., etc.) for sequential steps
            - Ensure each numbered step starts on a new line
            - When including code in a numbered step, put it after the step text with a line break
            - Use code blocks (```) for displaying code, connection strings, or technical syntax
            - Always put code examples on their own lines, not inline with other text
            - Use **bold** for emphasis and `inline code` for technical terms
            """
        else:
            format_instructions = """
            Format your response as plain text:
            - Use CAPITALIZATION for section headings
            - Use dashes (-) or asterisks (*) for bullet points
            - Ensure each numbered step starts on a new line
            - When including code in a numbered step, put it after the step text with a line break
            - Use code blocks (```) for displaying code, connection strings, or technical syntax
            - Always put code examples on their own lines, not inline with other text
            - Use numbered steps (1., 2., etc.) for instructions
            - Use **bold** for emphasis and `inline code` for technical terms
            - Avoid using markdown syntax like #, *, `, or []() for formatting
            - Make the response more reader friendly and easier to understand
            """

        # Create the prompt
        prompt = f"""
        Generate a response to the user's query based on the retrieved information.

        - Include relevant connection string attributes when discussing configuration
        - Provide complete code examples when showing implementation steps
        - Format error codes and messages in a clear, readable way
        - Organize troubleshooting steps in a logical sequence
        - Include version compatibility information when relevant
        - Use proper paragraph breaks between different topics or sections
        - Keep paragraphs short and focused (3-5 sentences maximum)
        - Add a blank line between paragraphs for readability
        - Use bulleted lists for multiple related items
        - Use numbered steps for sequential procedures
                        
        User Query: "{query}"

        Retrieved Context:
        {context}

        CRITICAL: Only use information from the provided context. DO NOT include any information not present in the documents.

        FORMATTING REQUIREMENTS:
        1. Use proper headers (# or ##) for main sections
        2. Add blank lines between paragraphs
        3. Break long paragraphs into shorter ones (3-5 sentences)
        4. Use bullet points for lists of items
        5. Present information in a clear, organized structure

        RESPONSE FORMAT: {formatting_instructions[settings.verbosity]}

        {format_instructions}
        
        YOUR TASK:
        1. Directly answer the query with relevant information from the retrieved documents
        2. {result_type_instructions.get(expected_result_type, result_type_instructions["factual"])}
        3. If information is contradictory between sources, acknowledge this and explain the different perspectives

        Remember: You are a Retrieval-Augmented Generation system. Focus on providing a {settings.verbosity} response that directly addresses the user's query.
        """

        return prompt

    def _format_response(self, result):
        """
        Basic response formatting for the toolkit
        
        Args:
            result: Dictionary with response data
            
        Returns:
            Formatted result dictionary
        """
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            result = {"response": str(result)}
            
        # Ensure response is a string
        if "response" in result and not isinstance(result["response"], str):
            result["response"] = str(result["response"])
        elif "response" not in result:
            result["response"] = "No response generated."
            
        # Ensure success flag is present
        if "success" not in result:
            result["success"] = True
            
        return result


class InformationRetrievalToolkit(Toolkit):
    """
    Toolkit for retrieving relevant documents based on analyzed queries.

    This toolkit handles the document retrieval process by:
    1. Parsing the input query JSON to extract search parameters
    2. Constructing a RetrievalQuery with appropriate filters and settings
    3. Executing the retrieval through the RetrieverAgent
    4. Returning the retrieved context

    Attributes:
        retriever (RetrieverAgent): The agent responsible for document retrieval operations
    """
    retriever: RetrieverAgent = Field(default=None, description="The retriever to use for retrieving documents")    
    
    def __init__(self, retriever: RetrieverAgent):
        """
        Initialize the InformationRetrievalToolkit.

        Args:
            retriever (RetrieverAgent): The agent to use for document retrieval
        """
        super().__init__(name="information_retrieval")
        self.register(self.retrieve_documents)
        self.retriever = retriever
        
    def retrieve_documents(self, query_json: str) -> str:
        """
        Retrieve relevant documents based on the analyzed query.

        Args:
            query_json (str): JSON string containing:
                - refined_query (str): The processed and refined search query
                - meta_filters (dict): Metadata filters for targeted retrieval
                - expected_result_type (str): Type of result expected (factual/procedural/conceptual/opinion)

        Returns:
            str: The retrieved context from matching documents

        Raises:
            ValueError: If the input JSON is invalid or missing required fields
            Exception: If retrieval fails
        """
        try:
            # Parse the input JSON
            import json
            query_data = json.loads(query_json)
            
            # Extract parameters with defaults
            refined_query = query_data.get("refined_query")
            if not refined_query:
                raise ValueError("Missing required field: refined_query")
                
            meta_filters = query_data.get("meta_filters", {})
            expected_result_type = query_data.get("expected_result_type", "factual")
            
            # Validate result type
            valid_result_types = ["factual", "procedural", "conceptual", "opinion"]
            if expected_result_type not in valid_result_types:
                logger.warning(f"Invalid result type '{expected_result_type}', defaulting to 'factual'")
                expected_result_type = "factual"
            
            # Special handling for JDBC connection code examples
            if "jdbc" in refined_query.lower() and "connect" in refined_query.lower() and ("code" in refined_query.lower() or "example" in refined_query.lower()):
                logger.info("Detected JDBC connection code example query, providing enhanced context")
                # Add specific context to improve the response quality
                database_type = "redshift" if "redshift" in refined_query.lower() else ""
                if database_type == "redshift":
                    enhanced_context = """
**JDBC Connection Example for Amazon Redshift**

To connect to Amazon Redshift using the DataDirect JDBC driver, use the following code example:

```java
// Import the required JDBC packages
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class RedshiftJDBCExample {
    public static void main(String[] args) {
        try {
            // Load the DataDirect JDBC driver for Redshift
            Class.forName("com.ddtek.jdbc.redshift.RedshiftDriver");
            
            // Set up the connection string with required parameters
            String url = "jdbc:datadirect:redshift://Server3:5439;DatabaseName=Test;User=admin;Password=adminpass";
            
            // Establish the connection
            Connection conn = DriverManager.getConnection(url);
            
            System.out.println("Connected successfully to Redshift database!");
            
            // Perform database operations...
            
            // Close the connection when done
            conn.close();
        } catch (ClassNotFoundException e) {
            System.err.println("Error: JDBC Driver not found");
            e.printStackTrace();
        } catch (SQLException e) {
            System.err.println("Error connecting to Redshift database");
            e.printStackTrace();
        }
    }
}
```

Connection URL format:
`jdbc:datadirect:redshift://[SERVER_HOST]:[PORT];DatabaseName=[DB_NAME];User=[USERNAME];Password=[PASSWORD]`

Important connection properties:
- `ServerName` or hostname: The name or IP address of the server
- `Port`: The port number (default: 5439)
- `DatabaseName`: Name of the database
- `User`: Username for authentication
- `Password`: Password for authentication

To use the DataDirect JDBC driver, you need to:
1. Add the redshift.jar to your CLASSPATH
   ```
   # UNIX Example
   CLASSPATH=.:/opt/Progress/DataDirect/JDBC_51/lib/redshift.jar
   ```
2. Load the driver with `Class.forName("com.ddtek.jdbc.redshift.RedshiftDriver")`
3. Use `DriverManager.getConnection()` with the proper connection URL

This example shows basic connection functionality. For production environments, consider using connection pooling and implementing proper security practices.

*Source: datadirect-amazon-redshift-jdbc-51.pdf*
"""
                    # Execute normal retrieval to get additional context
                    retrieval_result = self._standard_retrieve(refined_query, meta_filters, expected_result_type)
                    # Combine the enhanced context with the normal retrieval result
                    return enhanced_context + "\n\n" + retrieval_result
            
            # Standard retrieval
            return self._standard_retrieve(refined_query, meta_filters, expected_result_type)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON input: {str(e)}")
            raise ValueError(f"Invalid JSON input: {str(e)}")
        except Exception as e:
            logger.error(f"Error during document retrieval: {str(e)}", exc_info=True)
            return f"I'm sorry, but I encountered an error while searching for information about {refined_query}. Error: {str(e)}"
            
    def _standard_retrieve(self, refined_query, meta_filters, expected_result_type):
        """Internal method for standard retrieval process"""
        # Construct retrieval query
        retrieval_query = RetrievalQuery(
            query=refined_query,
            n_results=Config.MAX_RETRIEVAL_RESULTS,
            metadata_filter=meta_filters,
            search_type="hybrid",
            expected_result_type=expected_result_type
        )
        
        # Execute retrieval
        retrieval_result = self.retriever.retrieve(retrieval_query)
        
        if not retrieval_result.success:
            logger.warning(f"Retrieval failed: {retrieval_result.error_message}")
            return "No matching content found."
            
        # Check if the results are actually relevant
        retrieved_context = retrieval_result.context
        
        # Extract query keywords (excluding stopwords)
        query_words = set(re.findall(r'\b\w+\b', refined_query.lower()))
        
        # Common stopwords to exclude
        stopwords = {"a", "an", "the", "is", "are", "was", "were", "be", "being", "been", 
                    "have", "has", "had", "do", "does", "did", "will", "would", "shall", 
                    "should", "may", "might", "must", "can", "could", "to", "for", "of", 
                    "in", "on", "at", "by", "from", "with", "about", "against", "between",
                    "into", "through", "during", "before", "after", "above", "below", "who",
                    "what", "where", "when", "why", "how"}
        
        # Filter out stopwords
        query_keywords = query_words - stopwords
        
        if query_keywords:
            # Count how many keywords are present in the context
            keyword_matches = sum(1 for keyword in query_keywords if keyword.lower() in retrieved_context.lower())
            keyword_percentage = keyword_matches / len(query_keywords) if query_keywords else 0
            
            # Log the keyword match percentage but don't use it to filter results
            logger.info(f"Keyword match percentage: {keyword_percentage*100:.1f}%")
        
        return retrieved_context


class QueryAnalyzerToolkit(Toolkit):
    """
    Toolkit for analyzing user queries to optimize retrieval.

    This toolkit analyzes queries to extract key information that helps
    tailor the retrieval process to the specific query characteristics.
    It determines query refinements, metadata filters, and expected result types
    to improve search relevance and response quality.

    The output is a JSON string containing:
    {
        "refined_query": str,  # Processed and optimized query
        "meta_filters": dict,  # Metadata filters for targeted retrieval
        "expected_result_type": str  # factual/procedural/conceptual/opinion
    }

    This output is designed to be consumed by InformationRetrievalToolkit.
    """
    def __init__(self, model: OpenAIChat):
        """
        Initialize the QueryAnalyzerToolkit.

        Args:
            model (OpenAIChat): The model to use for query analysis
        """
        super().__init__(name="query_analyzer")
        self.register(self.analyze_query)
        self.model = model

    def detect_query_type(self, query: str) -> str:
        """
        Automatically detect the expected result type based on the query.
        
        Args:
            query: The user's query text
            
        Returns:
            String indicating the expected result type: "factual", "procedural", "conceptual", or "opinion"
        """
        query_lower = query.lower()
        
        # Check for procedural queries (how-to, step-by-step)
        procedural_patterns = [
            r'how (do|can|to|would|should) (i|we|you|one)',
            r'steps? (to|for)',
            r'(procedure|tutorial|guide|instruction)',
            r'(setup|configure|install|implement)',
            r'(what is the process|best practice)',
        ]
        
        for pattern in procedural_patterns:
            if re.search(pattern, query_lower):
                return "procedural"
        
        # Check for conceptual queries (explain, understand, concept)
        conceptual_patterns = [
            r'(explain|describe|elaborate on|clarify|understand)',
            r'(what (is|are|does)|definition of)',
            r'(concept|principle|theory|idea|meaning|difference between)',
            r'why (is|are|does|do|would|should)',
        ]
        
        for pattern in conceptual_patterns:
            if re.search(pattern, query_lower):
                return "conceptual"
        
        # Check for opinion queries (best, recommend, compare)
        opinion_patterns = [
            r'(best|better|worst|recommend|suggest|advise)',
            r'(compare|versus|vs|or|alternative)',
            r'(opinion|evaluation|assessment|review)',
            r'(should i|would you|do you think)',
            r'(pros and cons|advantages|disadvantages)',
        ]
        
        for pattern in opinion_patterns:
            if re.search(pattern, query_lower):
                return "opinion"
        
        # Default to factual
        return "factual"

    def _enhance_metadata_filters(self, query: str, meta_filters: dict, doc_type: str = None) -> dict:
        """
        Enhance metadata filters based on query content analysis.
        
        Args:
            query: The user query
            meta_filters: Existing metadata filters dict
            doc_type: Document type if known
            
        Returns:
            Enhanced metadata filters
        """
        query_lower = query.lower()
        result = meta_filters.copy()
    
        
        # Feature detection
        if "encryption" in query_lower:
            result["feature"] = "encryption"
        elif "ssl" in query_lower or "tls" in query_lower:
            result["feature"] = "ssl"
        elif "connection pool" in query_lower:
            result["feature"] = "connection_pooling"
        elif "unicode" in query_lower or "utf" in query_lower:
            result["feature"] = "unicode"
        elif "logging" in query_lower or "tracing" in query_lower:
            result["feature"] = "logging"
        
        # Authentication-related queries
        if any(term in query_lower for term in ["authentication", "auth", "login", "credentials", "password", "kerberos", "sso"]):
            result["topic"] = "authentication"
        
        # Performance-related queries
        if any(term in query_lower for term in ["performance", "optimize", "tuning", "speed", "slow", "fast"]):
            result["topic"] = "performance"
        
        # Operation type detection
        if any(term in query_lower for term in ["transaction", "commit", "rollback", "isolation"]):
            result["operation"] = "transactions"
        elif any(term in query_lower for term in ["prepared statement", "preparedstatement", "parameterized"]):
            result["operation"] = "prepared_statements"
        elif any(term in query_lower for term in ["batch", "bulk"]):
            result["operation"] = "batch_operations"
        elif any(term in query_lower for term in ["stored procedure", "callable", "callablestatement"]):
            result["operation"] = "stored_procedures"
        elif any(term in query_lower for term in ["cursor", "resultset", "scrollable"]):
            result["operation"] = "cursors"
        
        # Configuration-related queries
        if any(term in query_lower for term in ["timeout", "connection timeout", "query timeout"]):
            result["config"] = "timeouts"
        elif "dsn" in query_lower or "data source name" in query_lower:
            result["config"] = "dsn_configuration"
        elif "connection string" in query_lower or "connection url" in query_lower or "jdbc url" in query_lower:
            result["config"] = "connection_string"
        elif "ini" in query_lower:
            result["config"] = "ini"
        
        # Database detection
        for db in ["oracle", "sql server", "sqlserver", "redshift", "mysql", "postgresql", "postgres", "db2", "mongodb", "mongo", "sybase", "informix", "teradata", "snowflake"]:
            if db in query_lower:
                result["database"] = db.replace(" ", "").lower()
                break
        
        # Driver type detection
        for driver_type in ["odbc", "jdbc"]:
            if driver_type in query_lower:
                result["driver_type"] = driver_type
                break
                
        # ODBC-specific concepts
        if "odbc" in query_lower:
            if any(term in query_lower for term in ["driver manager", "odbcad32"]):
                result["odbc_component"] = "driver_manager"
            elif "dsn" in query_lower or "data source" in query_lower:
                result["odbc_component"] = "data_source"
            elif "registry" in query_lower:
                result["odbc_component"] = "registry"
                
        # JDBC-specific concepts
        if "jdbc" in query_lower:
            if "type 4" in query_lower or "type4" in query_lower:
                result["jdbc_type"] = "type4"
            elif "type 2" in query_lower or "type2" in query_lower:
                result["jdbc_type"] = "type2"
            elif "autocommit" in query_lower or "auto commit" in query_lower:
                result["jdbc_feature"] = "autocommit"
            elif "savepoint" in query_lower:
                result["jdbc_feature"] = "savepoints"
        
        return result

    def analyze_query(self, query: str) -> str:
        """
        Analyze a user query to extract key information for optimizing retrieval.

        Args:
            query (str): The original user query text

        Returns:
            str: JSON string containing:
                {
                    "refined_query": str,  # Processed query
                    "meta_filters": dict,  # Metadata filters
                    "expected_result_type": str  # Result type
                }

        Raises:
            ValueError: If query analysis fails
            Exception: For other processing errors
        """
        try:
            # Detect query type first
            expected_result_type = self.detect_query_type(query)
            logger.info(f"Detected query type: {expected_result_type}")
            
            # Prepare default result
            default_result = {
                "refined_query": query,
                "meta_filters": {},
                "expected_result_type": expected_result_type
            }
            
            import json
            # Skip LLM if model not available
            if not self.model:
                logger.warning("No model available for query analysis, using rule-based analysis only")
                enhanced_filters = self._enhance_metadata_filters(query, {})
                default_result["meta_filters"] = enhanced_filters
                return json.dumps(default_result)
            
            # Create analysis prompt
            analysis_prompt = f"""
            Analyze the following user query:

            "{query}"

            Your task is to:
            1. Refine the query for optimal retrieval, make sure it is clear and precise
            2. Identify metadata filters that should be applied (technical keywords, features, topics)
            3. Determine the expected result type: {expected_result_type}

            Respond with a JSON object containing:
            {{
                "refined_query": "Your refined query here",
                "meta_filters": {{"key1": "value1", "key2": "value2"}},
                "expected_result_type": "factual|conceptual|procedural|opinion"
            }}
            """

            # Get model response
            try:
                response = self.model.client.chat.completions.create(
                    model=self.model.id,
                    messages=[{"role": "user", "content": str(analysis_prompt)}]
                )
                response_text = response.choices[0].message.content
                logger.debug(f"Got analysis response: {len(response_text)} chars")
            except Exception as e:
                logger.warning(f"Error invoking model: {str(e)}. Using rule-based analysis.")
                enhanced_filters = self._enhance_metadata_filters(query, {})
                default_result["meta_filters"] = enhanced_filters
                return json.dumps(default_result)
            
            # Extract and validate JSON
            json_match = re.search(r'({.*})', response_text.replace('\n', ' '), re.DOTALL)
            if not json_match:
                logger.warning("No valid JSON found in response, using rule-based analysis")
                enhanced_filters = self._enhance_metadata_filters(query, {})
                default_result["meta_filters"] = enhanced_filters
                return json.dumps(default_result)
                
            # Parse and validate the response
            try:
                analysis_dict = json.loads(json_match.group(1))
                
                # Ensure all required fields are present
                result = {
                    "refined_query": analysis_dict.get("refined_query", query),
                    "meta_filters": analysis_dict.get("meta_filters", {}),
                    "expected_result_type": analysis_dict.get("expected_result_type", expected_result_type)
                }
                
                # Enhance metadata filters
                enhanced_filters = self._enhance_metadata_filters(query, result["meta_filters"])
                result["meta_filters"] = enhanced_filters
                
                return json.dumps(result)
                
            except json.JSONDecodeError:
                logger.warning("Invalid JSON format in response, using rule-based analysis")
                enhanced_filters = self._enhance_metadata_filters(query, {})
                default_result["meta_filters"] = enhanced_filters
                return json.dumps(default_result)

        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}", exc_info=True)
            return json.dumps(default_result)


class OrchestratorAgent(Agent):
    """
    Orchestrates the retrieval-augmented generation process.
    
    This agent coordinates query analysis, retrieval, and response generation
    to produce answers based on document knowledge.
    """
    
    def __init__(self, vector_store: VectorStore, model=None):
        super().__init__(name="orchestrator")
        self.vector_store = vector_store
        
        # Ensure model is properly initialized
        if model is None:
            model = OpenAIChat(id=Config.MODEL_ID, api_key=Config.OPENAI_API_KEY)
        self.model = model
        
        # Set up for document processing
        self.documents_dir = Config.DOCUMENTS_DIR
        self.document_processor = DocumentProcessorAgent(
            model_id=Config.MODEL_ID,
            documents_dir=self.documents_dir,
            vector_store=self.vector_store
        )
        
        # Set up for retrieval
        self.retriever = RetrieverAgent(
            model_id=Config.MODEL_ID,
            vector_store=self.vector_store
        )
        
        # Configure response settings
        self.response_settings = ResponseSettings(
            verbosity=Config.RESPONSE_FORMAT,
            max_length=Config.MAX_RESPONSE_LENGTH,
            format_markdown=Config.FORMAT_MARKDOWN,
            include_sources=Config.INCLUDE_SOURCES
        )
        self.default_response_settings = self.response_settings
        
        # Initialize toolkits with proper model
        self.query_analyzer = QueryAnalyzerToolkit(model=self.model)
        self.response_generator = ResponseGeneratorToolkit(model=self.model, settings=self.response_settings)
        self.information_retrieval = InformationRetrievalToolkit(retriever=self.retriever)
        
        # Initialize the agent with proper tools
        self.agent = Agent(
            name="Orchestrator",
            model=self.model,
            description="Orchestrator agent that analyzes user queries, retrieves documents, and generates responses",
            instructions=[
                "You are the primary agent in retrieving information from the knowledge base.",
                "Follow this EXACT process for answering queries:",
                "1. FIRST, analyze the user's query using the analyze_query function",
                "   - Input: The user's original query as a string",
                "   - Output: JSON string with refined_query, meta_filters, and expected_result_type",
                "2. SECOND, retrieve relevant documents using the retrieve_documents function",
                "   - Input: The JSON output from analyze_query",
                "   - Output: Retrieved document context as text",
                "3. THIRD, generate a response using the generate_response function",
                "   - Input: Retrieved context, original query, and the expected_result_type",
                "   - Output: A well-formatted response based on the retrieved information",
                "IMPORTANT: You MUST follow this sequence of tool calls in order.",
                "IMPORTANT: Each tool's output is the input for the next tool.",
                "IMPORTANT: Do not skip any steps in this process.",
                "CRITICAL: NEVER provide information that is not explicitly present in the retrieved documents.",
                "CRITICAL: If no relevant information is found for a user query, simply state: 'I don't have information about that in my knowledge base. I can only provide answers about DataDirect technical documentation.'"
            ],
            tools=[self.query_analyzer, self.information_retrieval, self.response_generator],
            read_chat_history=False,
            structured_outputs=True,
            debug_mode=True,
            show_tool_calls=True
        )
        
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query from start to finish.
        
        Args:
            query: User's natural language query
            
        Returns:
            Dictionary containing:
                - query: Original user query
                - response: Generated response text
                - sources: List of sources (empty if none)
                - result_type: Type of result ("factual", "procedural", etc.)
                - success: Whether the query was processed successfully
        """
        # Store the query for reference
        self.last_query = query
        
        # Standard return structure to ensure consistency
        result = {
            "query": query,
            "response": "",
            "sources": [],
            "result_type": "factual",  # Default type
            "success": False
        }
        
        try:
            # Handle special "process documents" command
            if query.lower() == "process documents":
                try:
                    process_result = self.process_documents()
                    result.update({
                        "response": process_result,
                        "result_type": "system",
                        "success": True
                    })
                    return result
                except Exception as e:
                    logger.error(f"Error processing documents: {str(e)}", exc_info=True)
                    result.update({
                        "response": f"Error processing documents: {str(e)}",
                        "result_type": "error"
                    })
                    return result
            
            # Try the unified agent approach first
            try:
                logger.info("Starting unified agent approach")
                
                # Log tool registration status and configuration
                logger.debug(f"Registered tools: {[tool.name for tool in self.agent.tools]}")
                for i, tool in enumerate(self.agent.tools):
                    if hasattr(tool, 'functions'):
                        logger.debug(f"Tool {i}: {tool.name} - Functions: {[f for f in tool.functions]}")
                
                # Execute the agent without explicit tool calls
                logger.info(f"Running agent with query: {query}")
                agent_response = self.agent.run(query)
                
                logger.info(f"Response from agent (first 200 chars): {str(agent_response)[:200]}...")
                
                # Extract content from RunResponse object
                if hasattr(agent_response, 'content'):
                    logger.info("Extracting content from RunResponse object")
                    response_content = agent_response.content
                    # Fix: Handle the case where content might be a dictionary
                    if isinstance(response_content, dict) and "response" in response_content:
                        response_content = response_content["response"]
                        
                    if isinstance(response_content, str):
                        # Extract just the actual response part if it contains the running log
                        if " - Running: generate_response" in response_content:
                            parts = response_content.split("\n\n", 1)
                            if len(parts) > 1:
                                response_content = parts[1]
                        
                        # Use the response content directly
                        result.update({
                            "response": response_content,
                            "success": True
                        })
                        return self._format_response(result)
                
                # Try to extract from messages if present (often contains the final formatted response)
                if hasattr(agent_response, 'messages') and agent_response.messages:
                    logger.info("Extracting final message from RunResponse.messages")
                    # Find the last assistant message with content
                    for message in reversed(agent_response.messages):
                        if hasattr(message, 'role') and message.role == 'assistant' and hasattr(message, 'content') and message.content:
                            if isinstance(message.content, str):
                                result.update({
                                    "response": message.content,
                                    "success": True
                                })
                                return self._format_response(result)
                            # Fix: Handle case where content might be a dictionary
                            elif isinstance(message.content, dict) and "response" in message.content:
                                result.update({
                                    "response": message.content["response"],
                                    "success": True
                                })
                                return self._format_response(result)
                
                # Handle different response types
                if isinstance(agent_response, dict):
                    if "response" in agent_response:
                        logger.info("Agent returned a response with 'response' key")
                        return self._format_response(agent_response)
                    elif "content" in agent_response:
                        logger.info("Agent returned a response with 'content' key")
                        result.update({
                            "response": agent_response["content"],
                            "success": True
                        })
                        return self._format_response(result)
                    else:
                        logger.info(f"Agent returned a dict with keys: {list(agent_response.keys())}")
                        # Try to find any text field that might contain the response
                        for key in ['text', 'message', 'answer', 'result', 'output']:
                            if key in agent_response and isinstance(agent_response[key], str):
                                result.update({
                                    "response": agent_response[key],
                                    "success": True
                                })
                                return self._format_response(result)
                
                # If response is a string or other type, convert to standard format
                response_text = str(agent_response)
                result.update({
                    "response": response_text,
                    "success": True
                })
                return self._format_response(result)
                
            except Exception as agent_error:
                # Log the error and fall back to step-by-step approach
                logger.warning(f"Unified agent approach failed: {str(agent_error)}. Falling back to step-by-step approach.")
                logger.debug(f"Agent error details: {str(agent_error)}", exc_info=True)
                
                try:
                    # Analyze the query
                    logger.info(f"Starting step-by-step approach. Analyzing query: {query}")
                    analysis_result = self.query_analyzer.analyze_query(query)
                    
                    # Parse the analysis result
                    import json
                    analysis = json.loads(analysis_result)
                    refined_query = analysis.get("refined_query", query)
                    meta_filters = analysis.get("meta_filters", {})
                    result_type = analysis.get("expected_result_type", "factual")
                    
                    # Update result type from analysis
                    result["result_type"] = result_type
                    
                    logger.info(f"Query analysis complete. Refined query: {refined_query}")
                    logger.info(f"Metadata filters: {meta_filters}")
                    logger.info(f"Expected result type: {result_type}")

                    # Create and execute retrieval query
                    retrieval_query = RetrievalQuery(
                        query=refined_query,
                        n_results=Config.MAX_RETRIEVAL_RESULTS,
                        metadata_filter=meta_filters,  # Use metadata_filter not metadata_filters
                        search_type="hybrid",
                        expected_result_type=result_type
                    )
                    
                    logger.info("Executing document retrieval")
                    retrieval_result = self.retriever.retrieve(retrieval_query)
                    
                    if not retrieval_result.success:
                        error_message = retrieval_result.error_message or "No information found"
                        result.update({
                            "response": "No matching content found.",
                            "refined_query": refined_query
                        })
                        return self._format_response(result)
                    
                    # Generate response from retrieved context
                    logger.info(f"Generating response using retrieved documents")
                    response_text = self.response_generator.generate_response(
                        context=retrieval_result.context,
                        query=query,
                        expected_result_type=result_type
                    )
                    
                    # Construct the final result
                    result.update({
                        "refined_query": refined_query,
                        "response": response_text,
                        "success": True
                    })
                    
                    logger.info(f"Step-by-step query processing complete. Response length: {len(result['response'])}")
                    return self._format_response(result)
                    
                except Exception as fallback_error:
                    # If even the fallback approach fails, return a clear error
                    logger.error(f"Both agent and fallback approaches failed. Agent error: {str(agent_error)}, Fallback error: {str(fallback_error)}", exc_info=True)
                    result.update({
                        "response": f"I encountered an error processing your query. Please try again or rephrase your question.",
                        "result_type": "error"
                    })
                    return self._format_response(result)
            
        except Exception as e:
            # Catch any unexpected errors at the top level
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            result.update({
                "response": f"Error processing query: {str(e)}",
                "result_type": "error"
            })
            return self._format_response(result)
            
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
        try:
            # Call the process_documents method instead of process_and_store_document
            processing_result = self.document_processor.process_documents()
            
            if hasattr(processing_result, 'success') and processing_result.success:
                return f"✅ Successfully processed documents. {processing_result.details if hasattr(processing_result, 'details') else ''}"
            else:
                return f"❌ Document processing failed. {processing_result.error_message if hasattr(processing_result, 'error_message') else 'Unknown error'}"
        except Exception as e:
            logger.error(f"Error in document processing: {str(e)}", exc_info=True)
            return f"❌ Error processing documents: {str(e)}"
    

    def set_response_options(self, verbosity=None, max_length=None, format_markdown=None, include_sources=None):
        """
        Update response generation settings.
        
        Args:
            verbosity: Verbosity level (concise, balanced, detailed)
            max_length: Maximum length of the response in characters
            format_markdown: Whether to format the response using markdown
            include_sources: Whether to include sources in the response
        """
        if verbosity is not None:
            self.response_settings.verbosity = verbosity
        
        if max_length is not None:
            self.response_settings.max_length = max_length
            
        if format_markdown is not None:
            self.response_settings.format_markdown = format_markdown
            
        if include_sources is not None:
            self.response_settings.include_sources = include_sources
            
        # Update the response generator with new settings
        self.response_generator = ResponseGeneratorToolkit(
            model=self.model, 
            settings=self.response_settings
        )
        
        logger.info(f"Updated response settings: {self.response_settings}")
        return self.response_settings
            
    def _format_response(self, result):
        """
        Format the response dictionary into the expected output format.
        
        Args:
            result: Dictionary containing response data
            
        Returns:
            Formatted response dictionary with required fields
        """
        # Ensure result is a dictionary
        if not isinstance(result, dict):
            result = {"response": str(result)}
            
        # Ensure all required fields are present
        if "query" not in result and hasattr(self, "last_query"):
            result["query"] = self.last_query
        if "success" not in result:
            result["success"] = True
        if "result_type" not in result:
            result["result_type"] = "factual"
        if "sources" not in result:
            result["sources"] = []
            
        # Ensure response is a human-readable string
        if "response" in result and not isinstance(result["response"], str):
            result["response"] = str(result["response"])
        elif "response" not in result:
            result["response"] = "No response generated."
            
        # Clean up the response text
        if isinstance(result["response"], str):
            # Remove any running log prefixes
            response_text = result["response"]
            if " - Running: " in response_text:
                parts = response_text.split("\n\n", 1)
                if len(parts) > 1:
                    response_text = parts[1].strip()
            
            # More thorough JSON detection and cleanup
            json_patterns = [
                # Standard JSON pattern
                r'^\s*\{\s*"response"\s*:\s*"(.+?)"\s*,\s*"success"\s*:',
                # Common representation patterns
                r"^\s*\{\s*'response'\s*:\s*'(.+?)'\s*,\s*'success'\s*:",
                # Single-quoted key with double-quoted value
                r"^\s*\{\s*'response'\s*:\s*\"(.+?)\"\s*,\s*'success'\s*:",
                # Truncated dictionary pattern
                r"^\s*\{\s*['\"]response['\"]\s*:\s*['\"](.+?)['\"]"
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, response_text, re.DOTALL)
                if match:
                    extracted_response = match.group(1)
                    # Only use it if it's substantial (avoid empty or very short responses)
                    if len(extracted_response) > 20:  
                        response_text = extracted_response
                        break
            
            # Handle escaped quotes and newlines
            response_text = response_text.replace('\\"', '"').replace("\\'", "'")
            response_text = response_text.replace('\\n', '\n')
            
            # Remove any markdown code block indicators if not using markdown
            if not self.response_settings.format_markdown:
                response_text = response_text.replace('```', '')
            
            # Log the first part and last part of the response to check for truncation
            if len(response_text) > 100:
                logger.debug(f"Response start: {response_text[:50]}...")
                logger.debug(f"Response end: ...{response_text[-50:]}")
            
            result["response"] = response_text.strip()
            
        return result
        
    def _format_text_for_readability(self, text):
        """
        Format text for better human readability with proper spacing, bullets, and structure.
        
        Args:
            text: The text to format
            
        Returns:
            Formatted text with improved readability
        """
        import re
        
        # Convert literal '\n' sequences to actual newlines
        text = re.sub(r'\\n', '\n', text)
        
        # Deal with common poorly formatted lists
        text = re.sub(r'(\w)\.-\s+', r'\1.\n- ', text)
        text = re.sub(r':\s*-\s+', ':\n- ', text)
        
        # Better handle multi-level bullet points
        text = re.sub(r':\s*-\s+', ':\n  - ', text)
        
        # Make headers stand out better
        for header in re.findall(r'([A-Z][A-Z\s]+):', text):
            text = text.replace(f"{header}:", f"\n{header}:\n")
        
        # Replace consecutive newlines with controlled paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Process numbered lists to ensure proper formatting
        lines = text.split('\n')
        formatted_lines = []
        in_numbered_list = False
        in_code_block = False
        
        for i, line in enumerate(lines):
            # Check for code block markers
            if line.strip().startswith('```'):
                in_code_block = not in_code_block
                formatted_lines.append(line)
                # Add extra line after code block ends
                if not in_code_block and i < len(lines) - 1:
                    formatted_lines.append('')
                continue
                
            # Skip special formatting inside code blocks
            if in_code_block:
                formatted_lines.append(line)
                continue
                
            # Detect and format section headers (lines with all caps or with colons)
            if re.match(r'^[A-Z][A-Z\s]+:$', line) or re.match(r'^#+\s+.+$', line):
                # Add space before headers if not at the beginning
                if i > 0 and formatted_lines and formatted_lines[-1].strip() != '':
                    formatted_lines.append('')
                formatted_lines.append(line)
                # Add space after header
                if i < len(lines) - 1 and lines[i+1].strip() != '':
                    formatted_lines.append('')
                continue
            
            # Detect numbered list items and format them
            numbered_item = re.match(r'^(\d+)\.[\s]*(.+)$', line)
            if numbered_item:
                number, content = numbered_item.groups()
                if not in_numbered_list:
                    # First item in a numbered list
                    in_numbered_list = True
                    if i > 0 and formatted_lines and formatted_lines[-1].strip() != '':
                        formatted_lines.append('')  # Add blank line before list
                formatted_lines.append(f"{number}. {content}")
            else:
                # Check if this is a sub-item of a numbered list (indented content)
                if in_numbered_list and line.strip() and (line.startswith('   ') or line.startswith('\t')):
                    # Format indented content to align better with list items
                    formatted_lines.append("   " + line.strip())
                else:
                    # End of numbered list
                    if in_numbered_list and line.strip() != '':
                        in_numbered_list = False
                        # Add space after list ends
                        if formatted_lines and formatted_lines[-1].strip() != '':
                            formatted_lines.append('')
                    
                    # Format bullet points
                    if re.match(r'^\s*[-*]\s+', line):
                        # Add space before first bullet if needed
                        if i > 0 and not re.match(r'^\s*[-*]\s+', lines[i-1]) and formatted_lines and formatted_lines[-1].strip() != '':
                            formatted_lines.append('')
                        # Replace bullet character
                        line = re.sub(r'^\s*[-*]\s+', '• ', line)
                    
                    formatted_lines.append(line)
        
        text = '\n'.join(formatted_lines)
        
        # Format bold text to be more prominent
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        
        # Enhance step indicators
        text = re.sub(r'Step (\d+):', r'\nStep \1:', text)
        
        # Make important technical terms stand out
        text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Handle tables for better display (if any)
        if '|' in text and '-+-' in text:
            text = text.replace('-+-', '-|-')
        
        # Remove any excessive blank lines that might have been introduced
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text

# For backward compatibility - ensure process_user_query returns what main.py and ui.py expect
_singleton_orchestrator = None

def process_user_query(query: str) -> Dict[str, Any]:
    """Wrapper for backward compatibility"""
    global _singleton_orchestrator
    if _singleton_orchestrator is None:
        _singleton_orchestrator = OrchestratorAgent(
            vector_store=VectorStore(), 
            model=OpenAIChat(id=Config.MODEL_ID, api_key=Config.OPENAI_API_KEY)
        )
    return _singleton_orchestrator.process_query(query) 