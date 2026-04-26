# =============================================================================
# utils/logger.py
# Centralised logging setup for the entire system
# =============================================================================

import logging
import os
from backend.config import LOG_LEVEL, LOG_FILE

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a named logger with both console and file handlers.
    All modules call get_logger(__name__) to get their own logger.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger   # Already configured — avoid duplicate handlers

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    try:
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        pass    # If file write fails, console logging still works

    return logger