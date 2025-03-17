import streamlit as st
import time
import re
from typing import List, Dict, Optional
import threading
import os

# Simple environment variable approach - less aggressive but won't break imports
os.environ["STREAMLIT_WATCHDOG_EXCLUDE_PATHS"] = "torch,transformers,openai,llama_index"

from src.utils.vector_store import VectorStore
from src.agents.orchestrator_agent import OrchestratorAgent
from src.utils.config import Config

# Page configuration
st.set_page_config(
    page_title="DD Doc Assist", 
    page_icon=None,
    layout="centered"
)

# Apply custom CSS
st.markdown("""
<style>
    body {
        background-color: #ffffff;
        color: #333333;
    }
    
    .main {
        max-width: 700px;
        margin: 0 auto;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    
    .user-message {
        background-color: #e6f2ff;
        border-left: 4px solid #1E88E5;
    }
    
    .assistant-message {
        background-color: #f0f0f0;
        border-left: 4px solid #43a047;
    }
    
    pre {
        background-color: #f8f9fa;
        padding: 12px;
        border-radius: 4px;
        overflow-x: auto;
    }
    
    code {
        color: #e83e8c;
        font-family: SFMono-Regular, Consolas, Liberation Mono, Menlo, monospace;
        word-wrap: break-word;
    }
    
    .app-header {
        text-align: center;
        margin-bottom: 2rem;
    }

    .app-title {
        color: #1E88E5;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    /* Hide the menu and footer */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    
    /* Override avatar styles */
    .stChatMessageAvatar {
        display: none !important;
    }
    
    /* Improve code blocks and technical content */
    .highlight {
        background-color: #f0f0f0;
        padding: 2px 4px;
        border-radius: 3px;
    }
    
    /* Format normal text better */
    p {
        margin-bottom: 12px;
        line-height: 1.6;
    }
    
    h2, h3, h4 {
        margin-top: 24px;
        margin-bottom: 16px;
    }
    
    ul, ol {
        margin-bottom: 16px;
        padding-left: 24px;
    }
    
    li {
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Set default configuration
Config.RESPONSE_FORMAT = "balanced"
Config.INCLUDE_CITATIONS = True
Config.INCLUDE_SOURCES = True
Config.CHUNKING_STRATEGY = "semantic"
Config.USE_RERANKING = True
Config.FORMAT_MARKDOWN = True

def format_response(text):
    """
    Format the RAG response text for better readability.
    """
    # Ensure we have proper line breaks between sections
    text = re.sub(r'(\d+\.)([A-Z])', r'\1 \2', text)  # Add space after numbered lists
    
    # Format code blocks and inline code properly
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    
    # Ensure proper paragraph breaks
    text = re.sub(r'(\w)\n(\w)', r'\1 \2', text)  # Replace single newlines within paragraphs with spaces
    text = re.sub(r'\n\s*\n', r'\n\n', text)  # Normalize multiple newlines to double newlines
    
    # Format technical terms
    text = re.sub(r'(EncryptionMethod)=(\d+)', r'<b>\1=\2</b>', text)
    text = re.sub(r'(EncryptionMethod|LogonID|Password)\b', r'<b>\1</b>', text)
    
    # Add proper heading for sections
    text = re.sub(r'(Key Details from the Documents:)', r'<h3>\1</h3>', text)
    text = re.sub(r'(Connection Configuration:)', r'<h3>\1</h3>', text)
    text = re.sub(r'(Notes:)', r'<h3>\1</h3>', text)
    
    # Format numbered lists nicely
    text = re.sub(r'(\d+\.\s)', r'<br>\1', text)
    
    return text

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm DD Doc Assist, your document assistant. How can I help you today?"}
    ]

if "orchestrator" not in st.session_state:
    # Initialize the vector store
    vector_store = VectorStore()
    
    # Initialize orchestrator agent
    st.session_state.orchestrator = OrchestratorAgent(
        vector_store=vector_store
    )

# Header
st.markdown("""
<div class="app-header">
    <div class="app-title">DD Doc Assist</div>
</div>
""", unsafe_allow_html=True)

# Define the streaming generator
def response_generator(query: str):
    """Generate a streaming response from the RAG system"""
    # Initial message to show we're working
    yield "Retrieving information... "
    
    # Get actual response from orchestrator
    full_response = st.session_state.orchestrator.process_query(query)
    
    # Process the response for better formatting
    formatted_response = format_response(full_response)
    
    # Clear the initial message by yielding an empty string with carriage return
    yield "\r" + " " * len("Retrieving information... ") + "\r"
    
    # Just return the full formatted response in one go
    yield formatted_response

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=None):
        if message["role"] == "assistant" and not message["content"].startswith("Hello"):
            # Use HTML for better formatting of assistant responses
            st.markdown(message["content"], unsafe_allow_html=True)
        else:
            st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask a question about your documents...")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user", avatar=None):
        st.markdown(prompt)
    
    # Display assistant message with streaming response
    with st.chat_message("assistant", avatar=None):
        response_container = st.empty()
        full_response = ""
        
        # Get streaming response
        for chunk in response_generator(prompt):
            if chunk.startswith("\r"):
                # This is a special control character to clear the line
                full_response = ""
                response_container.markdown(full_response)
            else:
                full_response = chunk
                response_container.markdown(full_response, unsafe_allow_html=True)
        
        # Add complete response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response}) 