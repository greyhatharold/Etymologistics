"""
Main entry point for the Etymologistics application.

This module initializes all components and starts the Streamlit interface.
"""

import os
import streamlit as st
from loguru import logger
from pathlib import Path

from src.gui import EtymologyUI
from src.rag import RAGPipeline
from src.utils.logging_config import setup_logging

def main():
    """Initialize and run the application."""
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(
        log_path=str(log_dir / "etymology.log"),
        console_level="INFO",
        file_level="DEBUG"
    )
    
    try:
        logger.info("Initializing application components...")
        
        # Initialize pipeline
        pipeline = RAGPipeline()
        logger.info("Pipeline initialized successfully")
        
        # Create and run UI
        ui = EtymologyUI(pipeline=pipeline)
        logger.info("UI initialized successfully")
        
        # Run Streamlit app
        ui.run()
        logger.info("Application running")
        
    except Exception as e:
        logger.error(f"Application initialization failed: {str(e)}")
        st.error("Failed to initialize application. Please check the logs for details.")

if __name__ == "__main__":
    main() 