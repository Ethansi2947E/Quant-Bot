# main.py -- Entry point for Trading Bot
"""
Main entry point for the trading bot system.
- Configures logging via config.py
- Loads environment and config
- Starts and stops the TradingBot
"""

import os
import asyncio
import logging
import traceback
from dotenv import load_dotenv

# --- Prevent __pycache__ creation ---
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

# --- Load environment and config ---
load_dotenv(override=True)
from config.config import (
    MT5_CONFIG,
    TRADING_CONFIG,
    TELEGRAM_CONFIG,
    LOG_CONFIG,
)
from src.utils.logging_setup import setup_logging

# --- Configure Logging ---
# All logging is now handled by the setup_logging function
setup_logging(LOG_CONFIG)

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