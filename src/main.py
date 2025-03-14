#!/usr/bin/env python3
import os
import argparse
import dotenv
import logging

from src.models.response_settings import ResponseSettings
from src.utils.vector_store import VectorStore, AgnoEmbeddingFunction
from src.agents.document_processor_agent import DocumentProcessorAgent
from src.agents.retriever_agent import RetrieverAgent
from src.agents.orchestrator_agent import OrchestratorAgent

# Load environment variables from .env file
dotenv.load_dotenv()

# Check if OpenAI API key is set
if not os.environ.get("OPENAI_API_KEY"):
    print("\nERROR: The OPENAI_API_KEY environment variable is not set.")
    print("Please set it in your .env file or environment variables to use this application.")
    print("Example: OPENAI_API_KEY=your-api-key-here\n")
    exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# Configure Agno logging - set to DEBUG for detailed logs
agno_logger = logging.getLogger("agno")
agno_logger.setLevel(logging.DEBUG)

def print_banner():
    """Print a welcome banner for the application."""
    print("\n" + "="*80)
    print(" "*30 + "MULTI-AGENT RAG SYSTEM")
    print("="*80)
    print("\nWelcome to the Advanced Multi-Agent RAG System!")
    print("This system uses three specialized agents to provide intelligent document retrieval:")
    print("  1. Orchestrator Agent: Coordinates tasks and manages user interaction")
    print("  2. Document Processor: Analyzes PDF documents and builds the knowledge base")
    print("  3. Retriever Agent: Searches for relevant information in the knowledge base")
    print("\nSpecial commands:")
    print("  - 'process documents': Index documents in the data/documents directory")
    print("  - 'set format <concise|balanced|detailed>': Change response format")
    print("  - 'set citations <on|off>': Toggle source citations")
    print("  - 'set sources <on|off>': Toggle sources section")
    print("  - 'set length <number>': Set maximum response length")
    print("  - 'help': Show this menu")
    print("  - 'exit' or 'quit': End the session")
    print("="*80 + "\n")

def main():
    """Main function to run the RAG system."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the RAG system")
    parser.add_argument("--format", choices=["concise", "balanced", "detailed"], 
                        default="balanced", help="Response format verbosity")
    parser.add_argument("--citations", action="store_true", 
                        help="Include citations in the response")
    parser.add_argument("--sources", action="store_true", 
                        help="Include a sources section in the response")
    parser.add_argument("--max_length", type=int, default=None, 
                        help="Maximum response length in characters")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug mode with more verbose logging")
    args = parser.parse_args()
    
    # Set logging level based on debug flag
    if args.debug:
        logger.setLevel(logging.DEBUG)
        agno_logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled with verbose logging")
    
    # Ensure directories exist
    os.makedirs("./data/documents", exist_ok=True)
    os.makedirs("./data/vector_store", exist_ok=True)
    
    # Set response settings
    response_settings = ResponseSettings(
        verbosity=args.format,
        include_citations=args.citations,
        include_sources_section=args.sources,
        max_length=args.max_length
    )
    
    # Log the response settings to verify they're being set correctly
    logger.info(f"Response settings: verbosity={response_settings.verbosity}, "
                f"citations={response_settings.include_citations}, "
                f"sources={response_settings.include_sources_section}, "
                f"max_length={response_settings.max_length}")
    
    # Initialize the vector store
    vector_store = VectorStore(
        persistence_directory="./data/vector_store",
        collection_name="document_store",
        embedding_function=AgnoEmbeddingFunction(id="text-embedding-3-small"),
        enable_caching=True
    )
    
    # Create the orchestrator agent
    orchestrator = OrchestratorAgent(
        model_id="gpt-4o",
        documents_dir="./data/documents",
        vector_store=vector_store,
        default_response_format=args.format
    )
    
    # Update orchestrator's response settings
    orchestrator.default_response_settings = response_settings
    
    # Print banner
    print_banner()
    
    # Run the interactive loop
    print("\n===== RAG System =====")
    print("Type 'quit' or 'exit' to end the session")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check if user wants to quit
        if user_input.lower() in ["quit", "exit"]:
            print("\nExiting...")
            break

        # Check for special formatting commands
        if user_input.lower().startswith("set format "):
            format_option = user_input.lower().split("set format ")[1].strip()
            if format_option in ["concise", "balanced", "detailed"]:
                orchestrator.default_response_settings.verbosity = format_option
                print(f"\nRAG: Response format set to '{format_option}'")
                continue
            else:
                print("\nRAG: Invalid format option. Use 'concise', 'balanced', or 'detailed'.")
                continue
        
        # Process the user input
        try:
            print("\nProcessing...")
            
            # Use the orchestrator's process_user_query method
            response = orchestrator.process_user_query(user_input)
            
            # Check if response is None or empty
            if not response:
                response = "No response generated. There might be an issue with the retrieval process."
            
            # Apply additional formatting based on response settings
            formatted_response = orchestrator.default_response_settings.format_response(response)
            
            # Display the response with clear formatting
            print(f"\nRAG [{orchestrator.default_response_settings.verbosity.upper()}]:")
            print(formatted_response)
            
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}", exc_info=True)
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 