"""
Main entry point for the ClipsAI demo application.
"""
import logging
import nltk
import os
from demo import create_demo

def initialize():
    """Initialize required resources."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    # Download NLTK data if needed (silently)
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.download('punkt')
        nltk.download('punkt_tab')
    except LookupError:
        nltk.download('punkt', quiet=True)

if __name__ == "__main__":
    data_store_dir = "/root/spill/data"
    initialize()
    demo = create_demo(data_store_dir)
    
    # Get server name from environment or use default
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    
    # Launch with proper configuration for sharing
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,  # Explicitly set port
        share=True,  # Enable sharing
        show_error=True,  # Show detailed errors
        show_api=False,  # Hide API docs
        quiet=False,  # Show startup messages
        allowed_paths=[data_store_dir]  # Allow access to our download directory
    )
