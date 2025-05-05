# Prevent Python from creating __pycache__ directories
import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import asyncio
import sys
import logging
import traceback
import MetaTrader5 as mt5
from dotenv import load_dotenv
from loguru import logger

# Initialize logging with console output only, no file logging
logging.basicConfig(
    level=logging.INFO,  # Set the default level to INFO
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)

# Make logging more prominent to ensure users notice this configuration
logging.info("============================================")
logging.info("LOGGING CONFIGURED WITH CONSOLE OUTPUT ONLY")
logging.info("NO LOG FILES WILL BE CREATED")
logging.info("============================================")

# Set log level for third-party libraries that may be too verbose
libraries_to_quiet = [
    'telegram', 
    'telegram.ext',
    'httpx', 
    'urllib3', 
    'requests'
]

for lib in libraries_to_quiet:
    lib_logger = logging.getLogger(lib)
    lib_logger.setLevel(logging.INFO)  # Set these loggers to INFO level by default

# Set log level for signal generators to reduce logs sent to Telegram
for logger_name in ['src.signal_generators']:
    module_logger = logging.getLogger(logger_name)
    module_logger.setLevel(logging.WARNING)  # Only WARNING and above will be sent to Telegram

# Edit this in main.py to set your strategy log level to INFO to see the INFO logs
for logger_name in ['src.strategy']:
    module_logger = logging.getLogger(logger_name)
    module_logger.setLevel(logging.DEBUG)  # Show INFO logs from strategy

# Create an interceptor to route standard logging messages to loguru
# This is critical for showing logs from modules that use standard logging
class InterceptHandler(logging.Handler):
    def emit(self, record):
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )

# Configure loguru
logger.remove()  # Remove default handlers
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",  # Changed from INFO to DEBUG
    colorize=True,
    # Remove enqueue parameter to avoid logging errors
    diagnose=False,  # Disable traceback to reduce log volume
    backtrace=False  # Disable backtrace to reduce log volume
)

# Add the interceptor to the standard logging system
logging.getLogger().handlers = [InterceptHandler()]

# Set the logging level for the root logger to match loguru
logging.getLogger().setLevel(logging.DEBUG)

# Import configurations (after DEFAULT_TIMEFRAME is modified)
from config.config import (
    MT5_CONFIG,
    TRADING_CONFIG,
    TELEGRAM_CONFIG,
    LOG_CONFIG,
)

# Override LOG_CONFIG to ensure no file logging
LOG_CONFIG.update({
    "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    "level": "DEBUG",
    # Remove file logging related settings
    "rotation": None,
    "retention": None,
    "compression": None,
    "use_file_logging": False,  # Add an explicit flag
    "colorize": True  # Ensure colors are enabled
})

# Import trading bot (after timeframe settings)
from src.trading_bot import TradingBot

async def main():
    """Main function to run the trading bot."""
    # Variables to track bot state
    trading_bot = None
    bot_running = False
    
    try:
        # Force reload environment variables to ensure we have the latest values
        load_dotenv(override=True)
        
        # Log the MT5 configuration being used
        logging.info(f"Using MT5 server: {MT5_CONFIG['server']}")
        logging.info(f"Using MT5 login: {MT5_CONFIG['login']}")
        
        # No shutdown method in MetaTrader5; nothing to do here
        logging.info("No MT5 shutdown method; skipping explicit shutdown.")
        
        # Create config object with all necessary configurations
        class Config:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)
                
            def __iter__(self):
                return iter(self.__dict__)
                
            def __getitem__(self, key):
                return self.__dict__[key]
                
            def get(self, key, default=None):
                return self.__dict__.get(key, default)
                
            def keys(self):
                return self.__dict__.keys()
                
            def values(self):
                return self.__dict__.values()
                
            def items(self):
                return self.__dict__.items()
                
            def update(self, other):
                if isinstance(other, dict):
                    self.__dict__.update(other)
                elif hasattr(other, '__dict__'):
                    self.__dict__.update(other.__dict__)
                    
        # Create config object with all necessary configurations
        config = Config(
            MT5_CONFIG=MT5_CONFIG,
            TRADING_CONFIG=TRADING_CONFIG,
            TELEGRAM_CONFIG=TELEGRAM_CONFIG,
            LOG_CONFIG=LOG_CONFIG
        )
        
        # Create trading bot instance
        trading_bot = TradingBot(config.__dict__)
        
        # Start the trading bot - now returns a future that completes when the bot should shut down
        logging.info("Starting trading bot...")
        try:
            shutdown_future = await trading_bot.start()
            
            if not isinstance(shutdown_future, asyncio.Future):
                logging.error(f"Trading bot failed to start properly - expected asyncio.Future but got {type(shutdown_future)}")
                if shutdown_future is None:
                    logging.error("shutdown_future is None - check if the trading_bot.start() method is returning a valid Future")
                return
                
            logging.info("Trading bot started successfully, waiting for completion")
            bot_running = True
                
            # Wait for the trading bot to signal it's done
            await shutdown_future
            logging.info("Trading bot signaled completion")
        except Exception as e:
            logging.error(f"Error during trading bot start: {str(e)}")
            logging.error(f"Detailed startup error: {traceback.format_exc()}")
            return
        
    except asyncio.CancelledError:
        logging.info("Trading bot task was cancelled")
        # Let the finally block handle cleanup
        
    except Exception as e:
        # Log any exceptions
        logging.error(f"Bot error: {str(e)}")
        logging.error(f"Detailed error trace: {traceback.format_exc()}")
        
    finally:
        # Ensure the bot is stopped
        if "trading_bot" in locals() and trading_bot is not None:
            try:
                logging.info("Stopping trading bot in finally block...")
                await trading_bot.stop()
                logging.info("Bot stopped")
            except Exception as e:
                logging.error(f"Error stopping bot: {str(e)}")
                logging.error(traceback.format_exc())
            
        logging.info("Bot shutdown complete")

if __name__ == "__main__":
    try:
        # Run the main coroutine
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Application stopped by user")
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}")
        logging.error(traceback.format_exc()) 