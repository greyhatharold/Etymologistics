"""
Logging configuration for the Etymologistics application.

This module sets up structured logging with both file and console outputs,
using different formats and levels for different handlers.
"""

import sys
from pathlib import Path
from loguru import logger
from typing import Dict, Any

# Default log format with colors for console
CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

# Detailed format for file logging
FILE_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss} | "
    "{level: <8} | "
    "{name}:{function}:{line} | "
    "process:{process} | "
    "thread:{thread} | "
    "{message}"
)

def setup_logging(
    log_path: str = "logs/etymology.log",
    console_level: str = "INFO",
    file_level: str = "DEBUG"
) -> None:
    """
    Configure logging with both console and file handlers.
    
    Args:
        log_path: Path to log file
        console_level: Minimum level for console output
        file_level: Minimum level for file output
    """
    # Create logs directory if it doesn't exist
    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with color formatting
    logger.add(
        sys.stdout,
        format=CONSOLE_FORMAT,
        level=console_level,
        colorize=True,
        backtrace=True,
        diagnose=True
    )
    
    # Add file handler with detailed formatting
    logger.add(
        log_path,
        format=FILE_FORMAT,
        level=file_level,
        rotation="1 day",
        retention="30 days",
        compression="gz",
        backtrace=True,
        diagnose=True
    )
    
    logger.info("Logging system initialized")

def get_logger(name: str) -> logger:
    """
    Get a logger instance with the given name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logger.bind(name=name)

def log_method_call(method_name: str, **kwargs: Any) -> None:
    """
    Log a method call with its arguments.
    
    Args:
        method_name: Name of the method being called
        **kwargs: Method arguments to log
    """
    logger.debug(
        f"Method call: {method_name}",
        method=method_name,
        arguments=kwargs
    )

def log_method_result(method_name: str, result: Any, duration: float) -> None:
    """
    Log a method's result and execution time.
    
    Args:
        method_name: Name of the method
        result: Method's return value
        duration: Execution time in seconds
    """
    logger.debug(
        f"Method result: {method_name}",
        method=method_name,
        result=result,
        duration=f"{duration:.3f}s"
    ) 