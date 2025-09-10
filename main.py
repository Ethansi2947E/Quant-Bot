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
from datetime import datetime
from dotenv import load_dotenv

# Diagnostic print to help identify initialization order
print("main.py starting - before any logging setup")

# --- Prevent __pycache__ creation ---
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

def display_banner():
    """Display the Qant Bot ASCII banner with colors."""
    # ANSI color codes
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    banner = f"""
{CYAN}╔══════════════════════════════════════════════════════════════════════════╗{RESET}
{CYAN}║                                                                          ║{RESET}
{CYAN}║{RESET}     {BOLD}{MAGENTA}  ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗    ██████╗  ██████╗ ████████╗{RESET} {CYAN}║{RESET}
{CYAN}║{RESET}     {BOLD}{MAGENTA} ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝    ██╔══██╗██╔═══██╗╚══██╔══╝{RESET} {CYAN}║{RESET}
{CYAN}║{RESET}     {BOLD}{CYAN} ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║       ██████╔╝██║   ██║   ██║{RESET}    {CYAN}║{RESET}
{CYAN}║{RESET}     {BOLD}{WHITE} ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║       ██╔══██╗██║   ██║   ██║{RESET}    {CYAN}║{RESET}
{CYAN}║{RESET}     {BOLD}{YELLOW} ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║       ██████╔╝╚██████╔╝   ██║{RESET}    {CYAN}║{RESET}
{CYAN}║{RESET}     {BOLD}{YELLOW}  ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝       ╚═════╝  ╚═════╝    ╚═╝{RESET}    {CYAN}║{RESET}
{CYAN}║                                                                          ║{RESET}
{CYAN}║{RESET}                    {BOLD}{GREEN}🚀 ADVANCED AUTOMATED TRADING SYSTEM 🚀{RESET}                   {CYAN}║{RESET}
{CYAN}║                                                                          ║{RESET}
{CYAN}║{RESET}        {WHITE}┌─────────────────────────────────────────────────────────┐{RESET}        {CYAN}║{RESET}
{CYAN}║{RESET}        {WHITE}│{RESET} {BOLD}{CYAN}📈 Algorithmic Trading  {RESET}│{RESET} {BOLD}{MAGENTA}💰 Risk Management{RESET}     {WHITE}│{RESET}        {CYAN}║{RESET}
{CYAN}║{RESET}        {WHITE}│{RESET} {BOLD}{YELLOW}⚡ Real-time Analysis   {RESET}│{RESET} {BOLD}{GREEN}📊 Market Intelligence{RESET} {WHITE}│{RESET}        {CYAN}║{RESET}
{CYAN}║{RESET}        {WHITE}└─────────────────────────────────────────────────────────┘{RESET}        {CYAN}║{RESET}
{CYAN}║                                                                          ║{RESET}
{CYAN}║{RESET}                          {BOLD}{WHITE}Developer: {CYAN}@dante_billz{RESET}                         {CYAN}║{RESET}
{CYAN}║{RESET}                    {WHITE}Telegram: {CYAN}https://t.me/dante_billz{RESET}                   {CYAN}║{RESET}
{CYAN}║                                                                          ║{RESET}
{CYAN}╚══════════════════════════════════════════════════════════════════════════╝{RESET}

{BOLD}{CYAN}[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]{RESET} {GREEN}System Status: {BOLD}INITIALIZING...{RESET}
{BOLD}{MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}
"""
    print(banner)

def display_startup_info():
    """Display startup information with styling."""
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    WHITE = '\033[97m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    print(f"{BOLD}{CYAN}🔧 SYSTEM INITIALIZATION{RESET}")
    print(f"{WHITE}├─{RESET} {GREEN}Loading Configuration...{RESET}")
    print(f"{WHITE}├─{RESET} {GREEN}Establishing MT5 Connection...{RESET}")
    print(f"{WHITE}├─{RESET} {GREEN}Initializing Trading Algorithms...{RESET}")
    print(f"{WHITE}├─{RESET} {GREEN}Setting up Telegram Integration...{RESET}")
    print(f"{WHITE}└─{RESET} {YELLOW}Ready for Trading Operations{RESET}")
    print(f"{BOLD}{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}\n")

# --- Load environment and config ---
load_dotenv(override=True)

# Display the banner first
display_banner()

from config.config import (
    MT5_CONFIG,
    TRADING_CONFIG,
    TELEGRAM_CONFIG,
    LOG_CONFIG,
)
from src.utils.logging_setup import setup_logging

# --- Configure Logging ---
print("About to call setup_logging...")
setup_logging(LOG_CONFIG)
print("Logging setup complete.")

# Display startup info
display_startup_info()

from src.trading_bot import TradingBot

async def main():
    """Main function to run the trading bot."""
    trading_bot = None
    try:
        # Success message with styling
        BOLD = '\033[1m'
        GREEN = '\033[92m'
        CYAN = '\033[96m'
        RESET = '\033[0m'
        
        logging.info(f"Using MT5 server: {MT5_CONFIG['server']}")
        logging.info(f"Using MT5 login: {MT5_CONFIG['login']}")
        
        print(f"\n{BOLD}{GREEN}🟢 QANT BOT ONLINE{RESET}")
        print(f"{CYAN}📡 Contact: @dante_billz on Telegram{RESET}")
        print(f"{CYAN}{'═' * 50}{RESET}\n")
        
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
        
        # Shutdown message
        RED = '\033[91m'
        BOLD = '\033[1m'
        RESET = '\033[0m'
        print(f"\n{BOLD}{RED}🔴 QANT BOT SHUTDOWN{RESET}")
        print(f"{RED}Bot operations terminated - @dante_billz{RESET}")
        logging.info("Bot shutdown complete")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Application stopped by user")
        print(f"\n\033[93m⚠️  Manual shutdown initiated by user\033[0m")
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}")
        logging.error(traceback.format_exc())