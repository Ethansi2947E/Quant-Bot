import asyncio
import sys
from loguru import logger
from config.config import *  # Import all config variables
import MetaTrader5 as mt5
import uvicorn
from src.api import app
from src.trading_bot import TradingBot

# Configure colored logging format
log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)

async def start_fastapi():
    """Start the FastAPI server."""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    # Configure logging
    logger.remove()  # Remove default handler
    
    # Add colored console logging
    logger.add(
        sys.stdout,
        format=log_format,
        level="DEBUG",
        colorize=True
    )
    
    # Add file logging
    logger.add(
        "logs/trading_bot.log",
        format=log_format,
        level="DEBUG",
        rotation="1 day",
        retention="1 month",
        compression="zip"
    )
    
    try:
        # Create config object with all necessary configurations
        config = type('Config', (), {
            'MT5_CONFIG': MT5_CONFIG,
            'TRADING_CONFIG': TRADING_CONFIG,
            'TELEGRAM_CONFIG': TELEGRAM_CONFIG,
            'AI_CONFIG': AI_CONFIG,
            'DB_CONFIG': DB_CONFIG,
            'LOG_CONFIG': LOG_CONFIG
        })
        
        # Start both the FastAPI server and the trading bot
        await asyncio.gather(
            start_fastapi(),
            TradingBot(config).start()
        )
        
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot error: {str(e)}")
    finally:
        # Ensure MT5 is properly shut down
        if mt5.initialize():
            mt5.shutdown()
        logger.info("Bot shutdown complete")

if __name__ == "__main__":
    asyncio.run(main()) 