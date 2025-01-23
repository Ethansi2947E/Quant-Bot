import sqlite3
from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime
from loguru import logger

class TradingDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.setup_database()
        
    def setup_database(self):
        """Create database tables if they don't exist."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Price data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume INTEGER NOT NULL,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            """)
            
            # POI data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS poi_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    type TEXT NOT NULL,
                    price_start REAL NOT NULL,
                    price_end REAL NOT NULL,
                    time DATETIME NOT NULL,
                    strength REAL NOT NULL,
                    status TEXT NOT NULL,
                    volume_imbalance REAL NOT NULL,
                    delta REAL NOT NULL
                )
            """)
            
            # Trading signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss REAL,
                    take_profit REAL,
                    reason TEXT
                )
            """)
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry_time DATETIME NOT NULL,
                    exit_time DATETIME,
                    trade_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    volume REAL NOT NULL,
                    pnl REAL,
                    status TEXT NOT NULL
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error setting up database: {str(e)}")
    
    def store_price_data(self, df: pd.DataFrame, symbol: str, timeframe: str):
        """Store price data in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Prepare data for insertion
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            # Insert data
            df.to_sql('price_data', conn, if_exists='append', index=False)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing price data: {str(e)}")
    
    def store_poi(self, poi_data: Dict):
        """Store POI data in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO poi_data (
                    symbol, timeframe, type, price_start, price_end,
                    time, strength, status, volume_imbalance, delta
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                poi_data['symbol'],
                poi_data['timeframe'],
                poi_data['type'],
                poi_data['price_start'],
                poi_data['price_end'],
                poi_data['time'],
                poi_data['strength'],
                poi_data['status'],
                poi_data['volume_imbalance'],
                poi_data['delta']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing POI data: {str(e)}")
    
    def store_signal(self, signal_data: Dict):
        """Store trading signal in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trading_signals (
                    symbol, timeframe, timestamp, signal_type,
                    confidence, entry_price, stop_loss, take_profit, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_data['symbol'],
                signal_data['timeframe'],
                signal_data['timestamp'],
                signal_data['signal_type'],
                signal_data['confidence'],
                signal_data['entry_price'],
                signal_data.get('stop_loss'),
                signal_data.get('take_profit'),
                signal_data.get('reason')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing signal data: {str(e)}")
    
    def store_trade(self, trade_data: Dict):
        """Store trade data in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trades (
                    symbol, entry_time, exit_time, trade_type,
                    entry_price, exit_price, volume, pnl, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_data['symbol'],
                trade_data['entry_time'],
                trade_data.get('exit_time'),
                trade_data['trade_type'],
                trade_data['entry_price'],
                trade_data.get('exit_price'),
                trade_data['volume'],
                trade_data.get('pnl'),
                trade_data['status']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing trade data: {str(e)}")
    
    def get_active_pois(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Get active POIs for a symbol and timeframe."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT *
                FROM poi_data
                WHERE symbol = ? AND timeframe = ? AND status = 'active'
                ORDER BY strength DESC
            """
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
            conn.close()
            return df
            
        except Exception as e:
            logger.error(f"Error getting active POIs: {str(e)}")
            return pd.DataFrame()
    
    def get_recent_signals(self, symbol: str, limit: int = 10) -> pd.DataFrame:
        """Get recent trading signals for a symbol."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"""
                SELECT *
                FROM trading_signals
                WHERE symbol = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(symbol, limit))
            conn.close()
            return df
            
        except Exception as e:
            logger.error(f"Error getting recent signals: {str(e)}")
            return pd.DataFrame()
    
    def get_trading_stats(self, symbol: str) -> Dict:
        """Get trading statistics for a symbol."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get total trades
            cursor.execute("""
                SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END)
                FROM trades
                WHERE symbol = ? AND status = 'closed'
            """, (symbol,))
            total_trades, winning_trades = cursor.fetchone()
            
            # Get profit metrics
            cursor.execute("""
                SELECT SUM(pnl), AVG(pnl), MAX(pnl), MIN(pnl)
                FROM trades
                WHERE symbol = ? AND status = 'closed'
            """, (symbol,))
            total_pnl, avg_pnl, max_pnl, min_pnl = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_trades': total_trades or 0,
                'winning_trades': winning_trades or 0,
                'win_rate': (winning_trades / total_trades * 100) if total_trades else 0,
                'total_pnl': total_pnl or 0,
                'average_pnl': avg_pnl or 0,
                'max_pnl': max_pnl or 0,
                'min_pnl': min_pnl or 0
            }
            
        except Exception as e:
            logger.error(f"Error getting trading stats: {str(e)}")
            return {} 