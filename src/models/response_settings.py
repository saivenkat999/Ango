from typing import Optional, Literal
from pydantic import BaseModel, Field
import logging

# Configure logger for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ResponseSettings")

ResponseFormat = Literal["concise", "balanced", "detailed"]

class ResponseSettings(BaseModel):
    """Settings for customizing response format and content."""
    verbosity: ResponseFormat = Field(
        default="balanced", 
        description="How detailed the response should be"
    )
    include_citations: bool = Field(
        default=True, 
        description="Whether to include source citations"
    )
    include_sources_section: bool = Field(
        default=True, 
        description="Whether to include a dedicated sources section"
    )
    max_length: Optional[int] = Field(
        default=None, 
        description="Maximum length of the response in characters (approximate)"
    )
    format_markdown: bool = Field(
        default=True, 
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
        
        # Handle Agno RunResponse objects properly
        # First check for the get_content_as_string method which is the most reliable
        if hasattr(response, 'get_content_as_string') and callable(getattr(response, 'get_content_as_string')):
            try:
                response = response.get_content_as_string()
            except Exception as e:
                logger.error(f"Error getting content as string: {str(e)}")
                # Fall back to content attribute
                if hasattr(response, 'content') and response.content is not None:
                    response = str(response.content)
        # Fall back to content attribute if get_content_as_string is not available
        elif hasattr(response, 'content') and response.content is not None:
            if isinstance(response.content, str):
                response = response.content
            else:
                response = str(response.content)
        
        # Ensure response is a string
        if not isinstance(response, str):
            # Try to convert to string as last resort
            try:
                response = str(response)
            except Exception as e:
                return f"Error formatting response: {str(e)}"
        
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
            
            # Removed the 150-character limit for concise responses
        
        # Apply max_length constraint if specified
        if self.max_length and len(response) > self.max_length:
            response = response[:self.max_length-3] + "..."
            
        return response 