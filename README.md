# RAG System

A retrieval-augmented generation (RAG) system that provides improved question answering capabilities by leveraging your documents.

## Configuration

The system is configured through a `.env` file. You can create one by copying the provided example:

```bash
cp .env.example .env
```

Then edit the `.env` file to set your configuration:

### Configuration Options

#### API Keys
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `COHERE_API_KEY`: Your Cohere API key (required for reranking)
- `LLAMA_CLOUD_API_KEY`: Your Llama Cloud API key (optional)

#### Vector Store Configuration
- `PERSISTENCE_DIRECTORY`: Directory for vector database storage (default: `./data/vector_db`)
- `COLLECTION_NAME`: Name of the vector collection (default: `documents`)
- `EMBEDDING_MODEL_ID`: Model ID for embeddings (default: `text-embedding-ada-002`)
- `SEARCH_TYPE`: Type of search to perform (default: `hybrid`)
- `BATCH_SIZE`: Batch size for document processing (default: `5`)

#### Document Processing Configuration
- `DOCUMENTS_DIR`: Directory containing documents to process (default: `./data/documents`)
- `CHUNKING_STRATEGY`: Strategy for chunking documents (default: `semantic`)
- `CHUNK_SIZE`: Maximum chunk size in tokens (default: `500`)
- `CHUNK_OVERLAP`: Overlap between chunks in tokens (default: `50`)

#### Response Settings
- `RESPONSE_FORMAT`: Format of the response (default: `balanced`)
- `MAX_RESPONSE_LENGTH`: Maximum length of the response (default: `1000`)
- `MAX_RESULTS`: Maximum number of results to return (default: `5`)
- `FORMAT_MARKDOWN`: Whether to format responses using markdown (default: `true`)

#### AI Model Configuration
- `MODEL_ID`: ID of the model to use (default: `gpt-3.5-turbo`)
- `USE_RERANKING`: Whether to use reranking (default: `true`)

#### Logging Configuration
- `LOG_LEVEL`: Logging level (default: `INFO`, options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)
- `LOG_TO_FILE`: Whether to log to a file (default: `false`)
- `LOG_FILE_PATH`: Path to the log file (default: `./logs/app.log`)
- `ENABLE_DETAILED_LOGGING`: Whether to use detailed logging format (default: `false`)
- `ENABLE_CONSOLE_LOGGING`: Whether to display logs in the console/terminal (default: `true`, set to `false` to disable all terminal logging)

## Running the System

1. Place your documents in the `DOCUMENTS_DIR` directory (default: `./data/documents`)
2. Run the system:

```bash
python src/main.py
```

## Usage

Once the system is running, you can:
1. Ask questions about your documents
2. Type 'exit' or 'quit' to end the session

The system will analyze your documents and provide answers to your questions based on the content of those documents. 