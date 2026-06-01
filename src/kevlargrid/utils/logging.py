"""Centralized thread-safe logging utility for KevlarGrid.

Configures colored ANSI stream formatting for standard console logs and
a rolling file handler targeting '.autosave/logs/vibedynalite.log'.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from typing import ClassVar

# Default parent logger name
LOGGER_NAME = "kevlargrid"

# Keep track of configuration state to prevent duplicate handler registration
_configured = False


class ColoredFormatter(logging.Formatter):
    """Custom logging formatter that applies ANSI colors to log levels."""

    COLORS: ClassVar[dict[int, str]] = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:

        color = self.COLORS.get(record.levelno, "")
        if color:
            orig_levelname = record.levelname
            record.levelname = f"{color}{orig_levelname}{self.RESET}"
            try:
                formatted = super().format(record)
            finally:
                record.levelname = orig_levelname
            return formatted
        return super().format(record)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure the root kevlargrid package logger with console and file handlers.

    Parameters
    ----------
    level : int
        Logging level (e.g., logging.INFO).
    """
    global _configured
    if _configured:
        return

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False  # Avoid duplicates if root logger also has handlers

    # Console Stream Handler with ANSI coloring
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Elegant console format showing time, level, source path with line number, and message
    console_format = "[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s"
    console_date_format = "%Y-%m-%d %H:%M:%S"

    console_formatter = ColoredFormatter(console_format, datefmt=console_date_format)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Rolling File Handler for persistent storage
    log_dir = ".autosave/logs"
    log_file = os.path.join(log_dir, "vibedynalite.log")

    try:
        os.makedirs(log_dir, exist_ok=True)
        # 5 MB per file, keep 5 backups
        file_handler = RotatingFileHandler(
            log_file, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setLevel(level)

        # File formatter (no colors for clean text files)
        file_format = (
            "[%(asctime)s] [%(levelname)s] [%(threadName)s] [%(filename)s:%(lineno)d]: %(message)s"
        )
        file_formatter = logging.Formatter(file_format, datefmt=console_date_format)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        # Fall back gracefully if file operations are restricted or fail
        logger.warning(f"Failed to initialize rotating file log handler: {e}")

    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Retrieve a logger scoped to the KevlarGrid subpackage.

    Parameters
    ----------
    name : str, optional
        Submodule name. If omitted, returns parent package logger.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    if not _configured:
        configure_logging()

    if name:
        return logging.getLogger(f"{LOGGER_NAME}.{name}")
    return logging.getLogger(LOGGER_NAME)
