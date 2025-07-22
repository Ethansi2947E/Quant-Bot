import os
import sys
import pandas as pd
from typing import Optional, Dict, Tuple
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError
from loguru import logger
from datetime import datetime

# Add project root to path to allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.mt5_handler import MT5Handler
from config.config import TRADING_CONFIG
# --- Import the single source of truth for the Trade model ---
from api.models import Trade

# --- Database Setup ---
DATABASE_URL = "sqlite:///trading_bot.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- ORM Model ---
# This local definition is removed to use the one from api.models

class DataManager:
    def __init__(self, mt5_handler: MT5Handler):
        # Database persistence attributes
        self.session = SessionLocal()
        self.mt5_handler = mt5_handler
        
        # Market data caching attributes
        self.market_data_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
        self.requirements: Dict[Tuple[str, str], int] = {}
        self.use_direct_fetch = True
        self.real_time_bars_count = 100

        self.init_db()

    def init_db(self):
        """
        Initializes the database and creates tables if they don't exist.
        """
        try:
            inspector = inspect(engine)
            # Import models here to ensure they are registered with Base
            from api.models import Signal
            
            # The Trade model is now imported directly, so we use its table definition
            if not inspector.has_table("trades"):
                logger.info("Creating 'trades' table in the database.")
                # We need to use the Base from api.models to create the table
                from api.database import Base as ApiBase
                ApiBase.metadata.create_all(bind=engine, tables=[Trade.__table__])
                logger.info("Table 'trades' created successfully.")

            if not inspector.has_table("signals"):
                logger.info("Creating 'signals' table in the database.")
                Base.metadata.create_all(bind=engine, tables=[Signal.__table__])
                logger.info("Table 'signals' created successfully.")
            else:
                logger.info("'trades' and 'signals' tables already exist.")
        except Exception as e:
            logger.error(f"An error occurred during database initialization: {e}")

    def get_market_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Retrieves market data from the cache for a given symbol and timeframe.
        """
        return self.market_data_cache.get((symbol, timeframe))

    def register_timeframe(self, symbol: str, timeframe: str, lookback: int):
        """Stores the data requirements for a symbol/timeframe pair."""
        self.requirements[(symbol, timeframe)] = lookback
        logger.info(f"Registered data requirement for {symbol}/{timeframe} with lookback {lookback}")

    def update_data(self, symbol: str, timeframe: str, force: bool = False, num_candles: Optional[int] = None):
        """Fetches market data from MT5 and updates the cache."""
        key = (symbol, timeframe)
        if num_candles is None:
            num_candles = self.requirements.get(key, 100)

        df = self.mt5_handler.get_market_data(symbol, timeframe, num_candles)
        if df is not None and not df.empty:
            self.market_data_cache[key] = df
            logger.debug(f"Updated data cache for {symbol}/{timeframe} with {len(df)} candles.")
        else:
            logger.warning(f"Failed to update data for {symbol}/{timeframe}.")

    def get_market_data_for_symbol(self, symbol: str, timeframes: list) -> dict:
        """Gets all cached data for a given symbol across multiple timeframes."""
        data = {}
        for tf in timeframes:
            key = (symbol, tf)
            if key in self.market_data_cache:
                data[tf] = self.market_data_cache[key]
        return data
    
    def synchronize_historical_trades(self):
        """
        Fetches all historical trades from MT5 and upserts them into the
        local SQLite database to ensure data consistency. This method will
        fetch all trades, ignoring the magic number.
        """
        logger.info("Starting historical trade synchronization for all account trades...")
        try:
            # We are fetching all history, so no magic number filter is applied here.
            historical_trades_df: Optional[pd.DataFrame] = self.mt5_handler.get_trade_history(days=90)

            if historical_trades_df is None or historical_trades_df.empty:
                logger.info("No historical trades found in MT5 for synchronization.")
                return

            upserted_count = 0
            for _, row in historical_trades_df.iterrows():
                ticket = row.get('ticket')
                if not ticket or pd.isna(ticket):
                    continue
                ticket = int(ticket)
                
                existing_trade = self.session.query(Trade).filter(Trade.ticket == ticket).first()

                if not existing_trade:
                    # The DataFrame now provides proper datetime objects
                    open_time = row.get('open_time')
                    close_time = row.get('close_time')

                    # Skip if essential times are missing
                    if pd.isna(open_time) or pd.isna(close_time):
                        logger.warning(f"Skipping trade with ticket {ticket} due to missing open/close time.")
                        continue
                    
                    new_trade = Trade(
                        ticket=ticket,
                        symbol=row.get('symbol'),
                        order_type=row.get('type'),
                        volume=row.get('volume'),
                        open_time=open_time,
                        open_price=row.get('open_price'),
                        close_time=close_time,
                        close_price=row.get('close_price'),
                        profit=row.get('profit'),
                        comment=row.get('comment')
                    )
                    self.session.add(new_trade)
                    upserted_count += 1
            
            if upserted_count > 0:
                self.session.commit()
                logger.info(f"Successfully synchronized and added {upserted_count} new historical trades to the database.")
            else:
                logger.info("Database is already up-to-date with historical trades.")

        except Exception as e:
            logger.error(f"An error occurred during historical trade synchronization: {e}")
            self.session.rollback()

    def log_trade(self, trade_data: dict):
        """
        Logs a single completed trade to the database.
        """
        ticket = trade_data.get('ticket')
        if not ticket:
            logger.error("Cannot log trade: ticket is missing.")
            return

        open_time_ts = trade_data.get('time_open')
        close_time_ts = trade_data.get('time_close')

        if open_time_ts is None or close_time_ts is None:
            logger.error(f"Cannot log trade {ticket}: open_time or close_time timestamp is missing.")
            return

        try:
            new_trade = Trade(
                ticket=ticket,
                symbol=trade_data.get('symbol'),
                order_type='buy' if trade_data.get('type') == 0 else 'sell',
                volume=trade_data.get('volume'),
                open_time=datetime.fromtimestamp(open_time_ts),
                open_price=trade_data.get('price_open'),
                close_time=datetime.fromtimestamp(close_time_ts),
                close_price=trade_data.get('price_close'),
                profit=trade_data.get('profit'),
                comment=trade_data.get('comment')
            )
            self.session.add(new_trade)
            self.session.commit()
            logger.info(f"Successfully logged trade {ticket} to the database.")
        except IntegrityError:
            self.session.rollback()
            logger.warning(f"Trade with ticket {ticket} already exists. Skipping.")
        except Exception as e:
            self.session.rollback()
            logger.error(f"Failed to log trade {ticket}: {e}")

    def log_signal(self, signal_data: dict):
        """Logs a new trading signal to the database."""
        from api.models import Signal # Import locally to prevent circular dependency on startup
        try:
            db_signal = Signal(
                symbol=signal_data['symbol'],
                timeframe=signal_data['timeframe'],
                strategy=signal_data['strategy'],
                direction=signal_data['direction'],
                price=signal_data['price'],
                details=signal_data.get('details') # Optional field
            )
            self.session.add(db_signal)
            self.session.commit()
            logger.info(f"Successfully logged signal for {signal_data['symbol']} to the database.")
        except Exception as e:
            logger.error(f"Failed to log signal for {signal_data.get('symbol', 'N/A')} to database: {e}")
            self.session.rollback()

    def __del__(self):
        self.session.close() 