# Advanced Multi-Agent RAG System

This project implements an advanced Retrieval Augmented Generation (RAG) system using multiple agents built with the Agno framework.

## System Architecture

The system contains three specialized agents:

1. **Orchestrator Agent**:
   - Processes user input and distributes tasks to other agents
   - Reformulates vague queries into clear instructions
   - Formats information retrieved from the knowledge base for user consumption

2. **Document Processor Agent**:
   - Processes PDF documents from a local folder
   - Extracts text and metadata from documents
   - Builds and maintains a knowledge base using ChromaDB

3. **Retriever Agent**:
   - Takes query instructions from the Orchestrator Agent
   - Searches the ChromaDB vector database for relevant content
   - Returns contextually appropriate results

## Setup and Installation

1. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables:
   - Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

3. Add your PDF documents to the `data/documents/` directory.

4. Run the system:
   ```
   python src/main.py
   ```

## Usage

Once running, interact with the system by typing your queries. The Orchestrator Agent will:
- Process your query
- Direct the appropriate agents to retrieve information
- Return formatted, contextually relevant responses

## Requirements

- Python 3.9+
- OpenAI API key
- Sufficient PDF documents for knowledge base creation 