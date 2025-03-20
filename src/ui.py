import streamlit as st
import re
import os
import time

# Simple environment variable approach - less aggressive but won't break imports
os.environ["STREAMLIT_WATCHDOG_EXCLUDE_PATHS"] = "torch,transformers,openai,llama_index"

from src.utils.vector_store import VectorStore
from src.agents.orchestrator_agent import OrchestratorAgent
from src.utils.config import Config
from agno.models.openai import OpenAIChat

# Page configuration
st.set_page_config(
    page_title="DataDirect Assist",
    page_icon="data/dd_logo.png",
    layout="centered",
)

# Use cached resource for the vector store to avoid reinitialization
@st.cache_resource
def get_vector_store():
    """Initialize and return the vector store."""
    return VectorStore()

# Use cached resource for the orchestrator to avoid reinitialization
@st.cache_resource
def get_orchestrator():
    """Initialize and return the orchestrator."""
    # Initialize the model
    model = OpenAIChat(id=Config.MODEL_ID, api_key=Config.OPENAI_API_KEY)
    return OrchestratorAgent(vector_store=get_vector_store(), model=model)

def clean_text_for_display(text):
    """Minimal cleaning of text for display"""
    # Remove any regex artifacts
    text = re.sub(r'\$\d+', '', text)
    
    # Fix headers by ensuring there's a space after # and newlines before/after
    text = re.sub(r'(?<!\n)(#+)([^ \n])', r'\n\1 \2', text)
    text = re.sub(r'(#+)[ ]+(.+?)[ ]*(?!\n)', r'\1 \2\n', text)
    
    # Fix code blocks that might be broken
    text = re.sub(r'```(\w*)\r?\n?\s*', r'```\1\n', text)
    text = re.sub(r'\s*\r?\n?```', r'\n```', text)
    
    # Fix inline code
    text = re.sub(r'`\s+', r'`', text)
    text = re.sub(r'\s+`', r'`', text)
    
    # CRITICAL FIX: Ensure numbered lists always have a line break before them, regardless of context
    # Look for pattern where a number with period follows text without a line break
    text = re.sub(r'([.!?:])(\s*)(\d+\.\s+)', r'\1\n\n\3', text)
    text = re.sub(r'([^.\n])(\s*)(\d+\.\s+)', r'\1\n\n\3', text)
    
    # Ensure numbered lists are properly formatted with newlines before them
    text = re.sub(r'(?<!\n)(\d+\.)[ ]+', r'\n\n\1 ', text)
    
    # Fix bullet points and numbered lists
    text = re.sub(r'^\*\s*(.+?)$', r'* \1', text, flags=re.MULTILINE)
    text = re.sub(r'^(\d+)\.\s*(.+?)$', r'\1. \2', text, flags=re.MULTILINE)
    
    # Add two newlines after colons that are likely to introduce lists
    text = re.sub(r'(:)(\s*)(\d+\.)', r'\1\n\n\3', text)
    
    return text

def display_formatted_content(text, container):
    """Display content with appropriate formatting using Streamlit's native components"""
    # Pre-process the text to ensure headers have proper spacing
    text = re.sub(r'(?<!\n)(#+)[ ]?(.+?)(?!\n)', r'\n\1 \2\n', text)
    
    # Ensure proper spacing for numbered lists
    text = re.sub(r'([.!?:])(\s*)(\d+\.\s+)', r'\1\n\n\3', text)
    
    # Extract headers and code blocks
    sections = []
    
    # Extract headers with their content - include those with no space after ###
    header_pattern = r'(?:^|\n)(#{1,3})[ ]?(.+?)(?:\n|$)'
    last_pos = 0
    
    for match in re.finditer(header_pattern, text, re.MULTILINE):
        # Add any content before this header
        if match.start() > last_pos:
            pre_content = text[last_pos:match.start()].strip()
            if pre_content:
                sections.append(("text", pre_content))
        
        # Add the header
        header_level = len(match.group(1))
        header_text = match.group(2).strip()
        
        # Find content for this header (until next header or end)
        next_match = re.search(header_pattern, text[match.end():], re.MULTILINE)
        if next_match:
            content_end = match.end() + next_match.start()
        else:
            content_end = len(text)
        
        header_content = text[match.end():content_end].strip()
        sections.append(("header", (header_level, header_text, header_content)))
        
        last_pos = content_end
    
    # Add any remaining content
    if last_pos < len(text):
        remaining = text[last_pos:].strip()
        if remaining:
            sections.append(("text", remaining))
    
    # If no headers found, just treat as a single text section
    if not sections:
        sections.append(("text", text))
    
    # Look for subheadings that might be missed by the main pattern
    for i, (section_type, content) in enumerate(sections):
        if section_type == "text":
            # Check for "### Header" patterns that might have been missed
            subheader_matches = re.finditer(r'(^|\n)(#{1,3})[ ]?(.+?)(?=\n|$)', content, re.MULTILINE)
            new_sections = []
            last_subheader_end = 0
            
            for submatch in subheader_matches:
                # Text before this subheader
                if submatch.start() > last_subheader_end:
                    pre_text = content[last_subheader_end:submatch.start()].strip()
                    if pre_text:
                        new_sections.append(("text", pre_text))
                
                # Add the subheader
                subheader_level = len(submatch.group(2))
                subheader_text = submatch.group(3).strip()
                
                # Find content for this subheader (until next subheader or end)
                next_submatch = re.search(r'(^|\n)(#{1,3})[ ]?(.+?)(?=\n|$)', 
                                         content[submatch.end():], re.MULTILINE)
                if next_submatch:
                    subcontent_end = submatch.end() + next_submatch.start()
                else:
                    subcontent_end = len(content)
                
                subheader_content = content[submatch.end():subcontent_end].strip()
                new_sections.append(("header", (subheader_level, subheader_text, subheader_content)))
                
                last_subheader_end = subcontent_end
            
            # Add any remaining content
            if last_subheader_end < len(content):
                remaining_text = content[last_subheader_end:].strip()
                if remaining_text:
                    new_sections.append(("text", remaining_text))
            
            # If subheaders were found, replace this section with the new sections
            if new_sections:
                sections[i:i+1] = new_sections
    
    # Process each section to extract numbered lists and code blocks
    for i, (section_type, content) in enumerate(sections):
        if section_type == "text":
            # Improved pattern for finding numbered lists - look for numbered lists even without newlines
            list_pattern = r'((?:^|\n|\s)(\d+\.)[ ]+.+?)(?=\n\d+\.[ ]+|\n\n|$)'
            list_matches = list(re.finditer(list_pattern, content, re.DOTALL | re.MULTILINE))
            
            if list_matches:
                new_parts = []
                last_pos = 0
                
                for list_match in list_matches:
                    # Get the full match and the specific numbered list marker
                    full_match = list_match.group(1)
                    list_marker = list_match.group(2)
                    match_start = list_match.start()
                    
                    # Handle case where the list item doesn't start with a newline
                    # Find where the actual numbered item starts
                    if not full_match.startswith('\n') and not full_match.startswith(list_marker):
                        # Find the position of the list marker
                        marker_pos = full_match.find(list_marker)
                        if marker_pos > 0:
                            # Adjust match_start to include text before the marker
                            adjusted_start = match_start + marker_pos
                            # Add text before the list marker to the pre_text
                            pre_text = content[last_pos:adjusted_start].strip()
                            if pre_text:
                                new_parts.append(("text", pre_text))
                            
                            # Extract just the numbered list item
                            list_text = full_match[marker_pos:].strip()
                            new_parts.append(("numbered_list", list_text))
                            last_pos = list_match.end()
                            continue
                    
                    # Text before this list item (normal case)
                    if match_start > last_pos:
                        pre_text = content[last_pos:match_start].strip()
                        if pre_text:
                            new_parts.append(("text", pre_text))
                    
                    # Add the list item as special list type
                    list_text = full_match.strip()
                    new_parts.append(("numbered_list", list_text))
                    
                    last_pos = list_match.end()
                
                # Add any remaining text
                if last_pos < len(content):
                    remaining = content[last_pos:].strip()
                    if remaining:
                        new_parts.append(("text", remaining))
                
                # Replace this section with the new parts
                if new_parts:
                    sections[i] = ("parts", new_parts)
    
    # Render each section
    for section_type, content in sections:
        if section_type == "text":
            # Clean any remaining "###" that might have been missed
            cleaned_content = re.sub(r'^#{1,3}\s+', '**', content, flags=re.MULTILINE)
            
            # Ensure numbered lists start on a new line with extra spacing
            cleaned_content = re.sub(r'(?<!\n)(\d+\.)[ ]+', r'\n\n\1 ', cleaned_content)
            
            # Extract code blocks to display them separately
            code_block_pattern = r'```(\w*)\n(.*?)\n```'
            code_blocks = list(re.finditer(code_block_pattern, cleaned_content, re.DOTALL))
            
            if code_blocks:
                last_pos = 0
                for code_match in code_blocks:
                    # Display text before code block
                    if code_match.start() > last_pos:
                        pre_code_text = cleaned_content[last_pos:code_match.start()].strip()
                        if pre_code_text:
                            container.markdown(pre_code_text)
                    
                    # Display code block
                    lang = code_match.group(1) or None
                    code = code_match.group(2)
                    container.code(code, language=lang)
                    
                    last_pos = code_match.end()
                
                # Display remaining text after last code block
                if last_pos < len(cleaned_content):
                    remaining = cleaned_content[last_pos:].strip()
                    if remaining:
                        container.markdown(remaining)
            else:
                container.markdown(cleaned_content)
                
        elif section_type == "header":
            level, title, header_content = content
            # Use appropriate header component
            if level == 1:
                container.header(title)
            elif level == 2:
                container.subheader(title)
            else:
                container.markdown(f"### {title}")  # Use markdown for level 3 headers
            
            # Display content under this header
            if header_content:
                # Look for code blocks in this section
                code_block_pattern = r'```(\w*)\n(.*?)\n```'
                
                # Replace code blocks with placeholders to avoid conflicts
                code_blocks = []
                
                for code_match in re.finditer(code_block_pattern, header_content, re.DOTALL):
                    lang = code_match.group(1) or None
                    code = code_match.group(2)
                    placeholder = f"__CODE_BLOCK_{len(code_blocks)}__"
                    code_blocks.append((placeholder, lang, code))
                    header_content = header_content.replace(code_match.group(0), placeholder)
                
                # Split by placeholders
                parts = []
                for part in re.split(r'(__CODE_BLOCK_\d+__)', header_content):
                    if part.startswith("__CODE_BLOCK_"):
                        # Find matching code block
                        for placeholder, lang, code in code_blocks:
                            if placeholder == part:
                                parts.append(("code", lang, code))
                                break
                    elif part.strip():
                        # Clean any remaining header markers
                        cleaned_part = re.sub(r'^#{1,3}\s+', '', part, flags=re.MULTILINE)
                        parts.append(("text", cleaned_part))
                
                # Display each part
                for part_type, *part_content in parts:
                    if part_type == "text":
                        # Ensure numbered lists start on a new line with extra spacing
                        text_content = re.sub(r'(?<!\n)(\d+\.)[ ]+', r'\n\n\1 ', part_content[0])
                        container.markdown(text_content)
                    elif part_type == "code":
                        lang, code = part_content
                        container.code(code, language=lang)
                        
        elif section_type == "parts":
            for part_type, part_content in content:
                if part_type == "text":
                    container.markdown(part_content)
                elif part_type == "numbered_list":
                    # Ensure the numbered list is properly formatted for markdown rendering
                    list_content = part_content.strip()
                    # Make sure it starts with a number and has proper spacing
                    if re.match(r'^\d+\.', list_content):
                        container.markdown(list_content)

def stream_response(response_text, speed=0.01):
    """
    Create a streaming generator for the response text
    
    Args:
        response_text: The full response text
        speed: Time delay between chunks in seconds
    
    Yields:
        Text chunks for streaming
    """
    # Stream by sentence to make the experience more natural
    sentences = re.split(r'([.!?]\s+)', response_text)
    
    # Recombine into proper sentences (the split keeps the delimiters)
    current_sentence = ""
    for i, part in enumerate(sentences):
        current_sentence += part
        
        # If this part is a delimiter (ends with . ! or ?) or we're at the end, yield the sentence
        if (i % 2 == 1) or (i == len(sentences) - 1 and current_sentence):
            words = current_sentence.split()
            
            # For very long sentences, chunk them further
            if len(words) > 10:
                chunks = [' '.join(words[i:i+5]) for i in range(0, len(words), 5)]
                for chunk in chunks:
                    yield chunk + " "
                    time.sleep(speed)
            else:
                yield current_sentence
                time.sleep(speed * 2)  # Slightly longer pause after full sentences
            
            current_sentence = ""

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! How can I help you today?"}
    ]

# Initialize the orchestrator with cached vector store
if "orchestrator" not in st.session_state:
    vector_store = get_vector_store()
    st.session_state.orchestrator = get_orchestrator()

# Header section
try:
    st.image("data/dd_logo.png", width=150)
except Exception:
    pass  # Silently fail if logo doesn't exist

st.title("DataDirect Assistant")
st.caption("Ask me anything about DataDirect")

# Sidebar with configuration options
with st.sidebar:
    st.header("Configuration")
    
    with st.expander("Advanced Settings"):
        Config.RESPONSE_FORMAT = st.selectbox(
            "Response Format", 
            ["concise", "balanced", "detailed"], 
            index=1
        )
        Config.INCLUDE_CITATIONS = st.checkbox("Include Citations", value=True)
        Config.INCLUDE_SOURCES = st.checkbox("Include Sources", value=True)
        Config.CHUNKING_STRATEGY = st.selectbox(
            "Chunking Strategy",
            ["semantic", "fixed", "paragraph"],
            index=0
        )
        Config.USE_RERANKING = st.checkbox("Use Reranking", value=True)
        Config.FORMAT_MARKDOWN = st.checkbox("Format as Markdown", value=True)
        
        # Apply formatting changes when settings change
        if "orchestrator" in st.session_state:
            st.session_state.orchestrator.set_response_options(
                verbosity=Config.RESPONSE_FORMAT,
                format_markdown=Config.FORMAT_MARKDOWN
            )
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! How can I help you today?"}
        ]
        st.rerun()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask a question about DataDirect...")

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        # Create a placeholder for the streaming response
        message_placeholder = st.empty()
        
        # Initial status
        status = st.status("Processing your query...", expanded=True)
        
        try:
            # Get response from orchestrator
            response_data = st.session_state.orchestrator.process_query(prompt)
            response_text = response_data.get("response", "No response generated.")
            result_type = response_data.get("result_type", "factual")
            
            # Update status
            status.update(label=f"Generating response...", state="running")
            
            # Stream the response
            full_response = ""
            for chunk in response_text.split(" "):
                full_response += chunk + " "
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)
            
            # Display final response
            message_placeholder.markdown(response_text)
            
            # Add response type indicator
            st.caption(f"Response type: {result_type.upper()}")
            
            # Add to chat history
            st.session_state.messages.append({"role": "assistant", "content": response_text})
            
            # Display sources if available and enabled
            if Config.INCLUDE_SOURCES and "sources" in response_data and response_data["sources"]:
                with st.expander("Sources"):
                    for i, source in enumerate(response_data["sources"]):
                        st.markdown(f"**{i+1}. {source.get('citation', 'Unknown source')}**")
            
            # Update status to complete
            status.update(label="Complete!", state="complete")
            
        except Exception as e:
            error_message = f"I'm sorry, I encountered an error: {str(e)}"
            message_placeholder.markdown(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            status.update(label="Error", state="error")