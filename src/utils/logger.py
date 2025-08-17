"""Logging utilities."""

import os
import time
import logging
import colorlog
from typing import Optional


def create_logger(
    name: str = 'GET_CAPTION',
    log_dir: str = 'logs',
    log_file: Optional[str] = None,
    level: str = 'INFO'
) -> logging.Logger:
    """Create a logger with both file and console handlers."""
    
    log_colors = {
        'DEBUG': 'blue',
        'INFO': 'white',
        'WARNING': 'green',
        'ERROR': 'red',
        'CRITICAL': 'yellow',
    }
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s [%(name)s] %(levelname)s: %(message)s%(reset)s",
        datefmt='%H:%M:%S',
        log_colors=log_colors
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        log_file = f"{name.lower()}_{timestamp}.log"
    
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(log_dir, log_file), 
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, level.upper()))
    file_formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


def setup_logging(config) -> logging.Logger:
    """Setup logging based on configuration."""
    return create_logger(
        name='GET_CAPTION',
        log_dir='logs',
        level=config.log_level
    )
