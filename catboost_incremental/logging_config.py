"""logging_config.py."""

import sys

from loguru import logger
from rich.logging import RichHandler


def setup_logger(level: str = "INFO"):
    """Set up a logger with RichHandler for better formatting and color support."""
    logger.add(sys.stderr, format="{message}")
    logger.configure(handlers=[{"sink": RichHandler(), "format": "{message}", "level": level}])
    return logger
