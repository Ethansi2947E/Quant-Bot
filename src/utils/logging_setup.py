# src/utils/logging_setup.py

import sys
import logging
from pathlib import Path
from loguru import logger

class InterceptHandler(logging.Handler):
    """Redirects standard logging messages to Loguru."""
    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        
        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
            
        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logging(config: dict):
    """Configures Loguru based on the provided config dictionary."""
    logger.remove()  # Remove default handler

    # Console logger
    logger.add(
        sys.stderr,
        level=config.get("level", "INFO"),
        format=config["format_console"],
        colorize=config.get("colorize", True),
        backtrace=config.get("backtrace", False),
        diagnose=config.get("diagnose", False)
    )

    # File logger (optional)
    if config.get("use_file_logging", False):
        log_file_path = config.get("log_file_path")
        if not isinstance(log_file_path, Path):
            raise ValueError("log_file_path must be a Path object in LOG_CONFIG.")
            
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_file_path,
            level=config.get("level", "INFO"),
            format=config["format_file"],
            rotation=config.get("rotation"),
            retention=config.get("retention"),
            compression=config.get("compression"),
            colorize=False, # File logs should not be colorized
            backtrace=config.get("backtrace", False),
            diagnose=config.get("diagnose", False)
        )

    # Intercept standard logging to ensure all logs go through Loguru
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.getLogger().setLevel(logging.INFO)
    logger.info("Logging configured via logging_setup.py.")
    if config.get('use_file_logging'):
        logger.info(f"File logging enabled: {config.get('log_file_path')}")
    else:
        logger.info("Console-only logging enabled.") 