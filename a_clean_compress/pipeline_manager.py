"""
Pipeline manager module responsible for orchestrating the data processing workflow.
Coordinates data ingestion, cleaning, and compression stages while providing logging
and error handling.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

from data_ingestion import load_from_local, load_from_url, DataSourceError
from data_cleaning import clean_data, DataCleaningError
from data_compression import compress_dataframe, verify_compression, CompressionError


class PipelineError(Exception):
    """Custom exception for pipeline-related errors."""
    pass


class PipelineManager:
    """
    Manages the execution of the data processing pipeline.
    
    Attributes:
        config: Configuration dictionary for the pipeline
        logger: Logger instance for tracking pipeline execution
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        log_file: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the pipeline manager.
        
        Args:
            config: Configuration dictionary containing pipeline settings
            log_file: Optional path to log file
            
        Raises:
            PipelineError: If configuration is invalid
        """
        self.config = self._validate_config(config)
        self.logger = self._setup_logging(log_file)
    
    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the pipeline configuration."""
        required_keys = {'input_path', 'output_path', 'compression_method'}
        missing_keys = required_keys - set(config.keys())
        
        if missing_keys:
            raise PipelineError(f"Missing required configuration keys: {missing_keys}")
        
        return config
    
    def _setup_logging(self, log_file: Optional[Union[str, Path]] = None) -> logging.Logger:
        """Configure logging for the pipeline."""
        logger = logging.getLogger('pipeline_manager')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Add console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Add file handler if log_file is specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _ingest_data(self):
        """Handle data ingestion stage."""
        self.logger.info("Starting data ingestion")
        
        try:
            input_path = self.config['input_path']
            
            # Handle URL or local file based on input path
            if input_path.startswith(('http://', 'https://')):
                df = load_from_url(input_path)
                self.logger.info(f"Successfully loaded data from URL: {input_path}")
            else:
                df = load_from_local(input_path)
                self.logger.info(f"Successfully loaded data from file: {input_path}")
            
            return df
        
        except DataSourceError as e:
            self.logger.error(f"Data ingestion failed: {str(e)}")
            raise PipelineError(f"Ingestion stage failed: {str(e)}")
    
    def _clean_data(self, df):
        """Handle data cleaning stage."""
        self.logger.info("Starting data cleaning")
        
        try:
            cleaning_config = self.config.get('cleaning_config', {})
            cleaned_df = clean_data(df, **cleaning_config)
            self.logger.info("Successfully cleaned data")
            return cleaned_df
        
        except DataCleaningError as e:
            self.logger.error(f"Data cleaning failed: {str(e)}")
            raise PipelineError(f"Cleaning stage failed: {str(e)}")
    
    def _compress_data(self, df):
        """Handle data compression stage."""
        self.logger.info("Starting data compression")
        
        try:
            output_path = Path(self.config['output_path'])
            method = self.config['compression_method']
            compression_level = self.config.get('compression_level')
            
            # Add timestamp to output filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = output_path.with_name(
                f"{output_path.stem}_{timestamp}{output_path.suffix}"
            )
            
            compressed_path = compress_dataframe(
                df,
                output_path,
                method=method,
                compression_level=compression_level
            )
            
            # Verify compression
            if verify_compression(df, compressed_path, method):
                self.logger.info(f"Successfully compressed data to: {compressed_path}")
                return compressed_path
            else:
                raise CompressionError("Compression verification failed")
        
        except CompressionError as e:
            self.logger.error(f"Data compression failed: {str(e)}")
            raise PipelineError(f"Compression stage failed: {str(e)}")
    
    def run_pipeline(self) -> Path:
        """
        Execute the complete data processing pipeline.
        
        Returns:
            Path to the final compressed output file
            
        Raises:
            PipelineError: If any stage of the pipeline fails
            
        Example:
            >>> config = {
            ...     'input_path': 'data/raw/input.csv',
            ...     'output_path': 'data/processed/output.csv',
            ...     'compression_method': 'gzip',
            ...     'cleaning_config': {
            ...         'date_columns': ['date'],
            ...         'numeric_columns': ['value']
            ...     }
            ... }
            >>> manager = PipelineManager(config)
            >>> output_path = manager.run_pipeline()
        """
        self.logger.info("Starting pipeline execution")
        
        try:
            # Stage 1: Data Ingestion
            df = self._ingest_data()
            
            # Stage 2: Data Cleaning
            cleaned_df = self._clean_data(df)
            
            # Stage 3: Data Compression
            output_path = self._compress_data(cleaned_df)
            
            self.logger.info("Pipeline execution completed successfully")
            return output_path
        
        except PipelineError as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in pipeline: {str(e)}")
            raise PipelineError(f"Pipeline failed with unexpected error: {str(e)}")


def run_pipeline(config: Dict[str, Any], log_file: Optional[Union[str, Path]] = None) -> Path:
    """
    Convenience function to run the pipeline with given configuration.
    
    Args:
        config: Pipeline configuration dictionary
        log_file: Optional path to log file
        
    Returns:
        Path to the final compressed output file
    """
    pipeline = PipelineManager(config, log_file)
    return pipeline.run_pipeline() 