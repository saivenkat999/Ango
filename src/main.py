#!/usr/bin/env python3
import os
import dotenv
import logging

from src.utils.vector_store import VectorStore
from src.agents.orchestrator_agent import OrchestratorAgent
from src.utils.config import Config

# Load environment variables from .env file
dotenv.load_dotenv()

# Logging is already configured by the Config class when it's imported
logger = logging.getLogger("Main")

# Configure Agno logging based on our configuration
agno_logger = logging.getLogger("agno")
if Config.ENABLE_DETAILED_LOGGING:
    agno_logger.setLevel(logging.DEBUG)
else:
    agno_logger.setLevel(logging.INFO)

def print_banner():
    """Print a welcome banner for the application."""
    print("\n" + "="*80)
    print(" "*25 + "ADVANCED MULTI-AGENT RAG SYSTEM")
    print("="*80)
    print("\nWelcome to the Advanced Multi-Agent RAG System with LlamaCloud, LanceDB, and Reranking!")
    print("This system uses three specialized agents to provide intelligent document retrieval:")
    print("  1. Orchestrator Agent: Coordinates tasks and manages user interaction")
    print("  2. Document Processor: Analyzes documents using LlamaCloud and Agno chunking")
    print("  3. Retriever Agent: Searches for relevant information with hybrid search and reranking")
    print("\nSpecial commands:")
    print("  - 'process documents': Index documents in the data/documents directory")
    print("  - 'set format <concise|balanced|detailed>': Change response format")
    print("  - 'set citations <on|off>': Toggle source citations")
    print("  - 'set sources <on|off>': Toggle sources section")
    print("  - 'set length <number>': Set maximum response length")
    print("  - 'set chunking <semantic|fixed_size|sentence|paragraph>': Change chunking strategy")
    print("  - 'set reranking <on|off>': Toggle result reranking")
    print("  - 'help': Show this menu")
    print("  - 'exit' or 'quit': End the session")
    print("="*80 + "\n")

def main():
    """Main function to initialize and run the RAG system."""
    
    # Print configuration at startup
    logger.info("Starting with configuration from .env file")
    Config.print_config()
    
    # Initialize the vector store using configuration from .env
    vector_store = VectorStore()
    
    # Initialize orchestrator with configuration
    orchestrator = OrchestratorAgent(
        vector_store=vector_store
    )
    
    # Show system introduction
    print("\nRAG System Initialized")
    print(f"Using model: {Config.MODEL_ID}")
    print(f"Document directory: {Config.DOCUMENTS_DIR}")
    print(f"Chunking strategy: {Config.CHUNKING_STRATEGY}")
    print(f"Response format: {Config.RESPONSE_FORMAT}")
    print(f"Reranking: {'ON' if Config.USE_RERANKING else 'OFF'}")
    print(f"Sources display: {'ON' if Config.INCLUDE_SOURCES else 'OFF'}")
    print(f"Logging level: {Config.LOG_LEVEL}" + (f" (to {Config.LOG_FILE_PATH})" if Config.LOG_TO_FILE else ""))
    print("\nType 'exit' or 'quit' to end the session")
    print("\n-----------------------------------------")
    
    # Main interaction loop
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Check if user wants to quit
        if user_input.lower() in ["quit", "exit"]:
            print("\nExiting...")
            break
            
        # Process the user's query using the orchestrator
        response = orchestrator.process_query(user_input)
        
        # Display the response
        print(f"\nRAG [{Config.RESPONSE_FORMAT.upper()}]: {response}")
        
if __name__ == "__main__":
    main() 