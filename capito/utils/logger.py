"""
Logging Utilities
================

Centralized logging configuration for the CAPito system.
"""

import logging
import sys
import os
from datetime import datetime
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> logging.Logger:
    """
    Setup centralized logging for CAPito.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file name
        log_dir: Directory for log files
        
    Returns:
        Configured logger
    """
    # Create logs directory
    if log_file:
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    logger = logging.getLogger('capito')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        if not log_file.endswith('.log'):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f"{log_file}_{timestamp}.log"
        
        file_path = os.path.join(log_dir, log_file)
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {file_path}")
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module."""
    return logging.getLogger(f'capito.{name}')


def set_log_level(level: str) -> None:
    """Change log level for all CAPito loggers."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    capito_logger = logging.getLogger('capito')
    capito_logger.setLevel(log_level)
    
    for handler in capito_logger.handlers:
        handler.setLevel(log_level)


def create_session_logger(session_name: str, log_dir: str = "logs") -> logging.Logger:
    """Create a logger for a specific processing session."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{session_name}_{timestamp}.log"
    
    return setup_logging(
        level="INFO",
        log_file=log_file,
        log_dir=log_dir
    )
