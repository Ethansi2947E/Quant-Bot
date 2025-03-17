import asyncio
import sys
import signal
from loguru import logger
from config.config import *  # Import all config variables
import MetaTrader5 as mt5
from src.trading_bot import TradingBot

# Configure colored logging format
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

async def main():
    # Configure logging
    # logger.remove()  # Remove default handler
    
    # Add colored console logging
    # logger.add(
    #     sys.stdout,
    #     format=log_format,
    #     level="DEBUG",
    #     colorize=True
    # )
    
    # Add file logging
    # logger.add(
    #     "logs/trading_bot.log",
    #     format=log_format,
    #     level="DEBUG",
    #     rotation="1 day",
    #     retention="1 month",
    #     compression="zip"
    # )
    
    # Create a variable to hold the trading bot instance so we can access it in finally block
    trading_bot = None
    
    try:
        # Create config object with all necessary configurations
        config = type('Config', (), {
            'MT5_CONFIG': MT5_CONFIG,
            'TRADING_CONFIG': TRADING_CONFIG,
            'TELEGRAM_CONFIG': TELEGRAM_CONFIG,
            'AI_CONFIG': AI_CONFIG,
            'LOG_CONFIG': LOG_CONFIG
        })
        
        # Create trading bot instance
        trading_bot = TradingBot(config)
        
        # Create tasks for each service
        trading_bot_task = asyncio.create_task(trading_bot.start())
        
        # Create an asyncio.Event that will be set when a shutdown signal is received
        shutdown_event = asyncio.Event()
        
        # Define a simple signal handler that sets the shutdown event
        def signal_handler():
            print("Received shutdown signal, stopping bot...")
            shutdown_event.set()
        
        # Register the signal handlers (SIGINT for CTRL+C, SIGTERM for termination) if supported
        loop = asyncio.get_running_loop()
        if sys.platform != "win32":
            loop.add_signal_handler(signal.SIGINT, signal_handler)
            loop.add_signal_handler(signal.SIGTERM, signal_handler)
        
        # Wait for the trading bot task to complete
        await trading_bot_task
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {str(e)}")
        logger.exception("Detailed error trace:")
    finally:
        # Properly shutdown the trading bot which will also shutdown the dashboard
        if trading_bot is not None:
            logger.info("Stopping trading bot and dashboard...")
            try:
                # Create a task to stop the trading bot and wait for it to complete
                stop_task = asyncio.create_task(trading_bot.stop())
                await asyncio.wait_for(stop_task, timeout=10.0)  # Set a timeout to avoid hanging
                logger.info("Trading bot and dashboard stopped successfully")
            except asyncio.TimeoutError:
                logger.warning("Timeout while stopping trading bot, forcing shutdown")
            except Exception as e:
                logger.error(f"Error stopping trading bot: {e}")
        
        # Ensure MT5 is properly shut down
        if mt5.initialize():
            mt5.shutdown()
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        logger.exception("Detailed error trace:") 