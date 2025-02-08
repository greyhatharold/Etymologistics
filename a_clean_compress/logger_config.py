"""
Logging configuration module that provides consistent logging setup across the application.
Centralizes log formatting, levels, and output handling.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict
from datetime import datetime


class LogConfigError(Exception):
    """Custom exception for logging configuration errors."""
    pass


def get_log_level(level: str) -> int:
    """
    Convert string log level to logging constant.
    
    Args:
        level: String representation of log level
        
    Returns:
        Logging level constant
        
    Raises:
        LogConfigError: If invalid log level provided
    """
    level_map: Dict[str, int] = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    normalized_level = level.upper()
    if normalized_level not in level_map:
        raise LogConfigError(
            f"Invalid log level: {level}. "
            f"Must be one of: {', '.join(level_map.keys())}"
        )
    
    return level_map[normalized_level]


def create_log_directory(log_path: Path) -> None:
    """
    Create directory for log file if it doesn't exist.
    
    Args:
        log_path: Path object representing log file location
        
    Raises:
        LogConfigError: If directory creation fails
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise LogConfigError(f"Failed to create log directory: {str(e)}")


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    module_name: str = "pipeline",
    format_string: Optional[str] = None,
    capture_warnings: bool = True
) -> logging.Logger:
    """
    Configure logging with consistent formatting and optional file output.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        module_name: Name of the module/application for the logger
        format_string: Optional custom log format string
        capture_warnings: Whether to capture Python warnings in logs
        
    Returns:
        Configured logger instance
        
    Raises:
        LogConfigError: If logging configuration fails
        
    Example:
        >>> logger = setup_logging(
        ...     level="INFO",
        ...     log_file="logs/pipeline.log",
        ...     module_name="data_pipeline"
        ... )
        >>> logger.info("Pipeline started")
    """
    try:
        # Convert level string to logging constant
        log_level = get_log_level(level)
        
        # Create logger
        logger = logging.getLogger(module_name)
        logger.setLevel(log_level)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        # Define log format
        if format_string is None:
            format_string = (
                "%(asctime)s - %(name)s - %(levelname)s - "
                "%(filename)s:%(lineno)d - %(message)s"
            )
        
        formatter = logging.Formatter(format_string)
        
        # Configure console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Configure file handler if specified
        if log_file:
            log_path = Path(log_file)
            
            # Create log directory if needed
            create_log_directory(log_path)
            
            # Add timestamp to log filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = log_path.with_name(
                f"{log_path.stem}_{timestamp}{log_path.suffix}"
            )
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Capture warnings if requested
        if capture_warnings:
            logging.captureWarnings(True)
            warnings_logger = logging.getLogger('py.warnings')
            warnings_logger.handlers = logger.handlers
        
        return logger
    
    except Exception as e:
        raise LogConfigError(f"Failed to configure logging: {str(e)}")


def get_module_logger(
    module_name: str,
    parent_logger: Optional[str] = "pipeline"
) -> logging.Logger:
    """
    Get a module-specific logger that inherits settings from the parent logger.
    
    Args:
        module_name: Name of the module requesting the logger
        parent_logger: Name of the parent logger to inherit from
        
    Returns:
        Logger instance for the module
        
    Example:
        >>> logger = get_module_logger("data_cleaning")
        >>> logger.info("Starting data cleaning")
    """
    return logging.getLogger(f"{parent_logger}.{module_name}") 