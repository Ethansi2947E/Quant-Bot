# main.py -- Entry point for Trading Bot
"""
Main entry point for the trading bot system.
- Configures logging (console only, no file logging)
- Loads environment and config
- Starts and stops the TradingBot
"""

import os
import sys
import asyncio
import logging
import traceback
from dotenv import load_dotenv
from loguru import logger

# --- Prevent __pycache__ creation ---
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# --- Configure standard logging (console only) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.info("============================================")
logging.info("LOGGING CONFIGURED WITH CONSOLE OUTPUT ONLY")
logging.info("NO LOG FILES WILL BE CREATED")
logging.info("============================================")



# --- Intercept standard logging to loguru ---
class InterceptHandler(logging.Handler):
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

logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="TRACE",
    colorize=True,
    diagnose=False,
    backtrace=False
)
logging.getLogger().handlers = [InterceptHandler()]
logging.getLogger().setLevel(logging.DEBUG)

# --- Load environment and config ---
load_dotenv(override=True)
from config.config import MT5_CONFIG, TRADING_CONFIG, TELEGRAM_CONFIG, LOG_CONFIG

# --- Override LOG_CONFIG for console-only logging ---
LOG_CONFIG.update({
    "format": "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    "level": "DEBUG",
    "rotation": None,
    "retention": None,
    "compression": None,
    "use_file_logging": False,
    "colorize": True
})

from src.trading_bot import TradingBot

async def main():
    """Main function to run the trading bot."""
    trading_bot = None
    try:
        logging.info(f"Using MT5 server: {MT5_CONFIG['server']}")
        logging.info(f"Using MT5 login: {MT5_CONFIG['login']}")
        # No explicit MT5 shutdown needed

        # Simple config object for TradingBot
        config = dict(
            MT5_CONFIG=MT5_CONFIG,
            TRADING_CONFIG=TRADING_CONFIG,
            TELEGRAM_CONFIG=TELEGRAM_CONFIG,
            LOG_CONFIG=LOG_CONFIG
        )
        trading_bot = TradingBot(config)
        logging.info("Starting trading bot...")
        shutdown_future = await trading_bot.start()
        if not isinstance(shutdown_future, asyncio.Future):
            logging.error(f"Trading bot failed to start properly - expected asyncio.Future but got {type(shutdown_future)}")
            return
        logging.info("Trading bot started successfully, waiting for completion")
        await shutdown_future
        logging.info("Trading bot signaled completion")
    except asyncio.CancelledError:
        logging.info("Trading bot task was cancelled")
    except Exception as e:
        logging.error(f"Bot error: {str(e)}")
        logging.error(f"Detailed error trace: {traceback.format_exc()}")
    finally:
        if trading_bot is not None:
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
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Application stopped by user")
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}")
        logging.error(traceback.format_exc()) 