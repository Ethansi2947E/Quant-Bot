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

    # Determine the log level from config, defaulting to INFO.
    log_level = config.get("level", "INFO").upper()
    
    # Print the log level being used (this will show in console output)
    print(f"Setting up logging with level: {log_level}")

    # Console logger
    logger.add(
        sys.stderr,
        level=log_level,
        format=config.get("format_console", "{message}"),
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
            level=log_level,
            format=config.get("format_file", "{message}"),
            rotation=config.get("rotation"),
            retention=config.get("retention"),
            compression=config.get("compression"),
            colorize=False,
            backtrace=config.get("backtrace", False),
            diagnose=config.get("diagnose", False)
        )

    # Intercept standard logging to ensure all logs go through Loguru.
    # This configuration forces the standard logging to respect our level.
    # Convert level name to numeric constant for the stdlib logger
    numeric_level = getattr(logging, log_level, logging.INFO)
    logging.basicConfig(handlers=[InterceptHandler()], level=numeric_level, force=True)
    
    logger.info("Logging configured via logging_setup.py.")
    logger.info(f"Log level set to '{log_level}'.") # Explicitly log the level being used.
    
    if config.get('use_file_logging'):
        logger.info(f"File logging enabled: {config.get('log_file_path')}")
    else:
        logger.info("Console-only logging enabled.")

    # Force all existing standard library loggers to respect the configured log level.
    for name, lg in logging.root.manager.loggerDict.items():
        if isinstance(lg, logging.Logger):
            lg.setLevel(max(lg.level, numeric_level))