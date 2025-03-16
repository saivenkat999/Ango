import os
import logging
from typing import Any, Dict, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration manager that loads settings from .env file"""
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    LOG_TO_FILE = os.getenv("LOG_TO_FILE", "false").lower() in ("true", "yes", "1", "t")
    LOG_FILE_PATH = os.getenv("LOG_FILE_PATH", "./logs/app.log") 
    ENABLE_DETAILED_LOGGING = os.getenv("ENABLE_DETAILED_LOGGING", "false").lower() in ("true", "yes", "1", "t")
    ENABLE_CONSOLE_LOGGING = os.getenv("ENABLE_CONSOLE_LOGGING", "true").lower() in ("true", "yes", "1", "t")
    
    # Vector store configuration
    PERSISTENCE_DIRECTORY = os.getenv("PERSISTENCE_DIRECTORY", "./data/vector_store")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "document_store")
    EMBEDDING_MODEL_ID = os.getenv("EMBEDDING_MODEL_ID", "BAAI/bge-small-en-v1.5")
    SEARCH_TYPE = os.getenv("SEARCH_TYPE", "hybrid")
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
    
    # Document processing configuration
    DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./data/documents")
    CHUNKING_STRATEGY = os.getenv("CHUNKING_STRATEGY", "recursive")
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "128"))
    
    # Response settings
    RESPONSE_FORMAT = os.getenv("RESPONSE_FORMAT", "balanced")  # concise, balanced, detailed
    MAX_RESPONSE_LENGTH = int(os.getenv("MAX_RESPONSE_LENGTH", "1500"))
    MAX_RESULTS = int(os.getenv("MAX_RESULTS", "4"))
    MAX_RETRIEVAL_RESULTS = int(os.getenv("MAX_RETRIEVAL_RESULTS", str(MAX_RESULTS)))
    FORMAT_MARKDOWN = os.getenv("FORMAT_MARKDOWN", "true").lower() in ("true", "yes", "1", "t")
    INCLUDE_SOURCES = os.getenv("INCLUDE_SOURCES", "true").lower() in ("true", "yes", "1", "t")
    
    # AI model configuration
    MODEL_ID = os.getenv("MODEL_ID", "gpt-4o")
    USE_RERANKING = os.getenv("USE_RERANKING", "true").lower() in ("true", "yes", "1", "t")
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY", "")
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Get all configuration values as a dictionary"""
        return {key: value for key, value in cls.__dict__.items() 
                if not key.startswith('__') and not callable(value)}
    
    @classmethod
    def print_config(cls, include_secrets: bool = False) -> None:
        """Print the current configuration (optionally hiding secrets)"""
        config = cls.get_config()
        
        # Remove API keys if include_secrets is False
        if not include_secrets:
            for key in list(config.keys()):
                if "API_KEY" in key:
                    config[key] = "********" if config[key] else ""
        
        logger.info("Current Configuration:")
        for key, value in config.items():
            if not key.startswith('__') and not callable(value):
                logger.info(f"  {key}: {value}")

    @classmethod
    def configure_logging(cls):
        """Configure the logging system based on the settings"""
        # Get the log level from the configuration
        log_level = getattr(logging, cls.LOG_LEVEL, logging.INFO)
        
        # Define log format - use a more detailed format if ENABLE_DETAILED_LOGGING is True
        if cls.ENABLE_DETAILED_LOGGING:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s'
        else:
            log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Create formatter
        formatter = logging.Formatter(log_format)
        
        # Configure the root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Clear existing handlers to avoid duplicate logs
        if root_logger.handlers:
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
        
        # Add a console handler if ENABLE_CONSOLE_LOGGING is True
        if cls.ENABLE_CONSOLE_LOGGING:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(log_level)
            root_logger.addHandler(console_handler)
        
        # Add a file handler if LOG_TO_FILE is True
        if cls.LOG_TO_FILE:
            # Ensure the log directory exists
            log_dir = os.path.dirname(cls.LOG_FILE_PATH)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Create and add the file handler
            file_handler = logging.FileHandler(cls.LOG_FILE_PATH)
            file_handler.setFormatter(formatter)
            file_handler.setLevel(log_level)
            root_logger.addHandler(file_handler)
            
            # Log that we're also logging to a file
            if cls.ENABLE_CONSOLE_LOGGING:
                logging.info(f"Logging to file: {cls.LOG_FILE_PATH}")
        
        # Only log this message if console logging is enabled
        if cls.ENABLE_CONSOLE_LOGGING:
            logging.info(f"Logging configured with level: {cls.LOG_LEVEL}")
        
        # Set specific logger levels for third-party libraries if needed
        if not cls.ENABLE_DETAILED_LOGGING:
            # Reduce verbosity for some chatty libraries
            logging.getLogger("urllib3").setLevel(logging.WARNING)
            logging.getLogger("requests").setLevel(logging.WARNING)
            
        # If no handlers were added, add a NullHandler to prevent "No handlers could be found" warnings
        if not root_logger.handlers:
            root_logger.addHandler(logging.NullHandler())

# Configure logging using our custom configuration
Config.configure_logging()

# Configure logger for this module after configuring the logging system
logger = logging.getLogger("Config")

# Print the configuration on module load (without secrets)
# Only print if console logging is enabled
if Config.ENABLE_CONSOLE_LOGGING:
    Config.print_config(include_secrets=False) 