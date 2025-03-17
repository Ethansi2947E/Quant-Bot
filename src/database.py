import sqlite3
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime, timedelta
import logging
from loguru import logger
import threading
import pandas as pd

class TradingDatabase:
    """
    Database handler for the trading bot.
    
    Handles storage of signals, trades, and performance metrics.
    Uses SQLite for persistence.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the database with the given path.
        
        Args:
            db_path: Optional path to the database file. If None, uses default path.
        """
        if db_path is None:
            # Create a database in the project directory
            base_dir = Path(__file__).resolve().parent.parent
            db_path = str(base_dir / "trading_data.db")
        
        self.db_path = db_path
        self.conn = None
        self.lock = threading.Lock()  # Add thread lock
        self.initialize_database()
    
    def initialize_database(self) -> None:
        """Initialize the database with required tables if they don't exist."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            cursor = self.conn.cursor()
            
            # Create signals table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                current_price REAL NOT NULL,
                confidence REAL NOT NULL,
                description TEXT,
                position_size REAL,
                risk_amount REAL,
                market_condition TEXT,
                volatility_state TEXT,
                session TEXT,
                htf_bias TEXT,
                raw_data TEXT,
                status TEXT DEFAULT 'active',
                executed INTEGER DEFAULT 0,
                execution_time TEXT
            )
            ''')
            
            # Create trades table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER,
                ticket TEXT,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL,
                stop_loss REAL,
                take_profit REAL,
                position_size REAL NOT NULL,
                open_time TEXT NOT NULL,
                close_time TEXT,
                profit_loss REAL,
                profit_loss_pips REAL,
                status TEXT NOT NULL,
                FOREIGN KEY (signal_id) REFERENCES signals (id)
            )
            ''')
            
            # Create performance metrics table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                losing_trades INTEGER NOT NULL,
                win_rate REAL NOT NULL,
                profit_loss REAL NOT NULL,
                profit_loss_percentage REAL NOT NULL,
                average_win REAL,
                average_loss REAL,
                largest_win REAL,
                largest_loss REAL,
                drawdown REAL,
                sharpe_ratio REAL,
                timeframe TEXT NOT NULL
            )
            ''')
            
            self.conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            if self.conn:
                self.conn.close()
                self.conn = None
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
    
    def insert_signal(self, signal: Dict[str, Any]) -> int:
        """
        Insert a new signal into the database.
        
        Args:
            signal: Dictionary containing signal data
            
        Returns:
            The ID of the inserted signal
        """
        try:
            if not self.conn:
                self.initialize_database()
            
            # Create a copy of the signal to avoid modifying the original
            signal_copy = signal.copy()
            
            # Convert Pandas Timestamp objects to ISO format strings
            for key, value in signal_copy.items():
                if isinstance(value, pd.Timestamp):
                    signal_copy[key] = value.isoformat()
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, pd.Timestamp):
                            value[sub_key] = sub_value.isoformat()
                elif isinstance(value, list):
                    for i, item in enumerate(value):
                        if isinstance(item, pd.Timestamp):
                            value[i] = item.isoformat()
                        elif isinstance(item, dict):
                            for sub_key, sub_value in item.items():
                                if isinstance(sub_value, pd.Timestamp):
                                    item[sub_key] = sub_value.isoformat()
            
            with self.lock:  # Use lock for thread safety
                cursor = self.conn.cursor()
                
                # Prepare take profit value - might be a complex structure in some signals
                take_profit = signal_copy.get('take_profit')
                if isinstance(take_profit, (list, dict)):
                    # If it's a complex structure, use the first TP level or a default
                    if isinstance(take_profit, list) and take_profit:
                        take_profit = take_profit[0]
                    elif isinstance(take_profit, dict) and 'level_1' in take_profit:
                        take_profit = take_profit['level_1']
                    else:
                        take_profit = signal_copy.get('entry_price', 0)
                
                # Store all signal data
                cursor.execute('''
                INSERT INTO signals (
                    timestamp, symbol, timeframe, strategy, direction, 
                    entry_price, stop_loss, take_profit, current_price, 
                    confidence, description, position_size, risk_amount,
                    market_condition, volatility_state, session, htf_bias, raw_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_copy.get('timestamp', datetime.now().isoformat()),
                    signal_copy.get('symbol', 'UNKNOWN'),
                    signal_copy.get('timeframe', ''),
                    signal_copy.get('strategy', ''),
                    signal_copy.get('direction', ''),
                    signal_copy.get('entry_price', 0.0),
                    signal_copy.get('stop_loss', 0.0),
                    take_profit,
                    signal_copy.get('current_price', 0.0),
                    signal_copy.get('confidence', 0.0),
                    signal_copy.get('description', ''),
                    signal_copy.get('position_size', 0.0),
                    signal_copy.get('risk_amount', 0.0),
                    signal_copy.get('market_condition', ''),
                    signal_copy.get('volatility_state', ''),
                    signal_copy.get('session', ''),
                    signal_copy.get('htf_bias', ''),
                    json.dumps(signal_copy, default=self._json_serializer)  # Use custom serializer
                ))
                
                self.conn.commit()
                signal_id = cursor.lastrowid
                logger.info(f"Signal inserted with ID: {signal_id}")
                return signal_id
        except Exception as e:
            logger.error(f"Error inserting signal: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return -1
    
    def _json_serializer(self, obj):
        """Custom JSON serializer to handle non-serializable objects."""
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def update_signal_status(self, signal_id: int, status: str, executed: bool = False) -> bool:
        """
        Update the status of a signal.
        
        Args:
            signal_id: The ID of the signal to update
            status: The new status of the signal
            executed: Whether the signal has been executed
            
        Returns:
            True if the update was successful, False otherwise
        """
        try:
            if not self.conn:
                self.initialize_database()
            
            cursor = self.conn.cursor()
            
            execution_time = datetime.now().isoformat() if executed else None
            
            cursor.execute(
                "UPDATE signals SET status = ?, executed = ?, execution_time = ? WHERE id = ?",
                (status, 1 if executed else 0, execution_time, signal_id)
            )
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating signal status: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def insert_trade(self, trade: Dict[str, Any], signal_id: Optional[int] = None) -> int:
        """
        Insert a new trade into the database.
        
        Args:
            trade: Dictionary containing trade data
            signal_id: Optional ID of the associated signal
            
        Returns:
            The ID of the inserted trade
        """
        try:
            if not self.conn:
                self.initialize_database()
            
            with self.lock:  # Use lock for thread safety
                cursor = self.conn.cursor()
                
                cursor.execute('''
                INSERT INTO trades (
                    signal_id, ticket, symbol, direction, entry_price,
                    current_price, stop_loss, take_profit, position_size,
                    open_time, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_id,
                    trade.get('ticket', ''),
                    trade.get('symbol', ''),
                    trade.get('direction', ''),
                    trade.get('entry_price', 0.0),
                    trade.get('current_price', trade.get('entry_price', 0.0)),
                    trade.get('stop_loss', 0.0),
                    trade.get('take_profit', 0.0),
                    trade.get('position_size', 0.0),
                    trade.get('open_time', datetime.now().isoformat()),
                    trade.get('status', 'open')
                ))
                
                self.conn.commit()
                trade_id = cursor.lastrowid
                logger.info(f"Trade inserted with ID: {trade_id}")
                
                # If this trade is associated with a signal, update the signal status
                if signal_id:
                    self.update_signal_status(signal_id, 'executed', True)
                
                return trade_id
        except Exception as e:
            logger.error(f"Error inserting trade: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return -1
    
    def update_trade(self, trade_id: int, trade_data: Dict[str, Any]) -> bool:
        """
        Update an existing trade in the database.
        
        Args:
            trade_id: The ID of the trade to update
            trade_data: Dictionary containing updated trade data
            
        Returns:
            True if the update was successful, False otherwise
        """
        try:
            if not self.conn:
                self.initialize_database()
            
            cursor = self.conn.cursor()
            
            # Build the SET clause dynamically based on provided data
            set_clauses = []
            params = []
            
            for key, value in trade_data.items():
                if key not in ['id', 'signal_id']:  # Don't update primary keys
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
            
            if not set_clauses:
                return False
            
            # Add the trade_id as the last parameter
            params.append(trade_id)
            
            # Execute the update
            cursor.execute(
                f"UPDATE trades SET {', '.join(set_clauses)} WHERE id = ?",
                params
            )
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error updating trade: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def close_trade(self, trade_id: int, close_price: float, profit_loss: float, 
                    profit_loss_pips: float, close_time: Optional[str] = None) -> bool:
        """
        Close a trade in the database.
        
        Args:
            trade_id: The ID of the trade to close
            close_price: The closing price of the trade
            profit_loss: The profit/loss amount
            profit_loss_pips: The profit/loss in pips
            close_time: The closing time, defaults to current time
            
        Returns:
            True if the closing was successful, False otherwise
        """
        try:
            if not self.conn:
                self.initialize_database()
            
            if close_time is None:
                close_time = datetime.now().isoformat()
            
            cursor = self.conn.cursor()
            
            cursor.execute(
                """
                UPDATE trades SET 
                    current_price = ?, 
                    close_time = ?, 
                    profit_loss = ?, 
                    profit_loss_pips = ?, 
                    status = 'closed' 
                WHERE id = ?
                """,
                (close_price, close_time, profit_loss, profit_loss_pips, trade_id)
            )
            
            self.conn.commit()
            return True
        except Exception as e:
            logger.error(f"Error closing trade: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return False
    
    def get_active_signals(self) -> List[Dict[str, Any]]:
        """
        Get all active signals from the database.
        
        Returns:
            List of dictionaries containing active signal data
        """
        try:
            if not self.conn:
                self.initialize_database()
            
            with self.lock:  # Use lock for thread safety
                cursor = self.conn.cursor()
                cursor.execute("SELECT * FROM signals WHERE status = 'active'")
                
                # Get column names
                columns = [desc[0] for desc in cursor.description]
                
                # Convert rows to dictionaries
                signals = []
                for row in cursor.fetchall():
                    signal_dict = dict(zip(columns, row))
                    # Parse raw_data JSON if available
                    if 'raw_data' in signal_dict and signal_dict['raw_data']:
                        try:
                            raw_data = json.loads(signal_dict['raw_data'])
                            # Merge raw_data with signal_dict, but don't overwrite existing keys
                            for key, value in raw_data.items():
                                if key not in signal_dict or signal_dict[key] is None:
                                    signal_dict[key] = value
                        except json.JSONDecodeError:
                            pass
                    signals.append(signal_dict)
                
                return signals
        except Exception as e:
            logger.error(f"Error getting active signals: {str(e)}")
            return []
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """
        Get all active trades from the database.
        
        Returns:
            List of dictionaries containing active trade data
        """
        try:
            if not self.conn:
                self.initialize_database()
            
            with self.lock:  # Use lock for thread safety
                cursor = self.conn.cursor()
                cursor.execute("SELECT * FROM trades WHERE status = 'open'")
                
                # Get column names
                columns = [desc[0] for desc in cursor.description]
                
                # Convert rows to dictionaries
                trades = []
                for row in cursor.fetchall():
                    trade_dict = dict(zip(columns, row))
                    trades.append(trade_dict)
                
                return trades
        except Exception as e:
            logger.error(f"Error getting active trades: {str(e)}")
            return []
    
    def get_trade_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get trade history from the database.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of dictionaries containing trade history data
        """
        try:
            if not self.conn:
                self.initialize_database()
            
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM trades ORDER BY open_time DESC LIMIT ?",
                (limit,)
            )
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Convert rows to dictionaries
            trades = []
            for row in cursor.fetchall():
                trade_dict = dict(zip(columns, row))
                trades.append(trade_dict)
            
            return trades
        except Exception as e:
            logger.error(f"Error getting trade history: {str(e)}")
            return []
    
    def get_signals_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Get signals within a date range.
        
        Args:
            start_date: Start date in ISO format
            end_date: End date in ISO format
            
        Returns:
            List of dictionaries containing signal data
        """
        try:
            if not self.conn:
                self.initialize_database()
            
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM signals WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp DESC",
                (start_date, end_date)
            )
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Convert rows to dictionaries
            signals = []
            for row in cursor.fetchall():
                signal_dict = dict(zip(columns, row))
                signals.append(signal_dict)
            
            return signals
        except Exception as e:
            logger.error(f"Error getting signals by date range: {str(e)}")
            return []
    
    def get_trades_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """
        Get trades within a date range.
        
        Args:
            start_date: Start date in ISO format
            end_date: End date in ISO format
            
        Returns:
            List of dictionaries containing trade data
        """
        try:
            if not self.conn:
                self.initialize_database()
            
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM trades WHERE open_time BETWEEN ? AND ? ORDER BY open_time DESC",
                (start_date, end_date)
            )
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Convert rows to dictionaries
            trades = []
            for row in cursor.fetchall():
                trade_dict = dict(zip(columns, row))
                trades.append(trade_dict)
            
            return trades
        except Exception as e:
            logger.error(f"Error getting trades by date range: {str(e)}")
            return []
    
    def update_or_insert_performance_metrics(self, metrics: Dict[str, Any]) -> int:
        """
        Update or insert performance metrics for a given date and timeframe.
        
        Args:
            metrics: Dictionary containing performance metrics
            
        Returns:
            The ID of the updated or inserted metrics record
        """
        try:
            if not self.conn:
                self.initialize_database()
            
            cursor = self.conn.cursor()
            
            # Check if we already have metrics for this date and timeframe
            date = metrics.get('date', datetime.now().strftime('%Y-%m-%d'))
            timeframe = metrics.get('timeframe', '1D')
            
            cursor.execute(
                "SELECT id FROM performance_metrics WHERE date = ? AND timeframe = ?",
                (date, timeframe)
            )
            
            existing_id = cursor.fetchone()
            
            if existing_id:
                # Update existing record
                set_clauses = []
                params = []
                
                for key, value in metrics.items():
                    if key not in ['id', 'date', 'timeframe']:  # Don't update primary keys or date/timeframe
                        set_clauses.append(f"{key} = ?")
                        params.append(value)
                
                if not set_clauses:
                    return existing_id[0]
                
                # Add date and timeframe as the last parameters
                params.extend([date, timeframe])
                
                # Execute the update
                cursor.execute(
                    f"UPDATE performance_metrics SET {', '.join(set_clauses)} WHERE date = ? AND timeframe = ?",
                    params
                )
                
                self.conn.commit()
                return existing_id[0]
            else:
                # Insert new record
                keys = list(metrics.keys())
                placeholders = ', '.join(['?'] * len(keys))
                values = [metrics[key] for key in keys]
                
                cursor.execute(
                    f"INSERT INTO performance_metrics ({', '.join(keys)}) VALUES ({placeholders})",
                    values
                )
                
                self.conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return -1
    
    def get_performance_metrics(self, timeframe: str = '1W', limit: int = 7) -> List[Dict[str, Any]]:
        """
        Get performance metrics for a given timeframe.
        
        Args:
            timeframe: The timeframe for which to get metrics
            limit: Maximum number of metric records to return
            
        Returns:
            List of dictionaries containing performance metrics
        """
        try:
            if not self.conn:
                self.initialize_database()
            
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM performance_metrics WHERE timeframe = ? ORDER BY date DESC LIMIT ?",
                (timeframe, limit)
            )
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Convert rows to dictionaries
            metrics = []
            for row in cursor.fetchall():
                metric_dict = dict(zip(columns, row))
                metrics.append(metric_dict)
            
            return metrics
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            return []
    
    def calculate_performance_metrics(self, timeframe: str = '1D') -> Dict[str, Any]:
        """
        Calculate performance metrics for closed trades within the specified timeframe.
        
        Args:
            timeframe: The timeframe for calculating metrics (1D, 1W, 1M, etc.)
            
        Returns:
            Dictionary containing calculated performance metrics
        """
        try:
            if not self.conn:
                self.initialize_database()
            
            # Define date range based on timeframe
            now = datetime.now()
            start_date = None
            
            if timeframe == '1D':
                start_date = datetime(now.year, now.month, now.day).isoformat()
            elif timeframe == '1W':
                # Start from the beginning of the current week (Monday)
                days_since_monday = now.weekday()
                start_date = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0).isoformat()
            elif timeframe == '1M':
                start_date = datetime(now.year, now.month, 1).isoformat()
            elif timeframe == '3M':
                # Start from 3 months ago
                month = now.month - 3
                year = now.year
                if month <= 0:
                    month += 12
                    year -= 1
                start_date = datetime(year, month, 1).isoformat()
            elif timeframe == '6M':
                # Start from 6 months ago
                month = now.month - 6
                year = now.year
                if month <= 0:
                    month += 12
                    year -= 1
                start_date = datetime(year, month, 1).isoformat()
            elif timeframe == '1Y':
                start_date = datetime(now.year - 1, now.month, now.day).isoformat()
            else:
                # Default to 1 week
                days_since_monday = now.weekday()
                start_date = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0).isoformat()
            
            # Get closed trades within the timeframe
            cursor = self.conn.cursor()
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN profit_loss <= 0 THEN 1 ELSE 0 END) as losing_trades,
                    SUM(profit_loss) as total_profit_loss,
                    AVG(CASE WHEN profit_loss > 0 THEN profit_loss ELSE NULL END) as average_win,
                    AVG(CASE WHEN profit_loss <= 0 THEN profit_loss ELSE NULL END) as average_loss,
                    MAX(CASE WHEN profit_loss > 0 THEN profit_loss ELSE 0 END) as largest_win,
                    MIN(CASE WHEN profit_loss <= 0 THEN profit_loss ELSE 0 END) as largest_loss
                FROM trades
                WHERE status = 'closed' AND close_time >= ?
                """,
                (start_date,)
            )
            
            result = cursor.fetchone()
            
            if not result or result[0] == 0:
                # No trades in the period
                return {
                    'date': now.strftime('%Y-%m-%d'),
                    'timeframe': timeframe,
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0.0,
                    'profit_loss': 0.0,
                    'profit_loss_percentage': 0.0,
                    'average_win': 0.0,
                    'average_loss': 0.0,
                    'largest_win': 0.0,
                    'largest_loss': 0.0,
                    'drawdown': 0.0,
                    'sharpe_ratio': 0.0
                }
            
            total_trades, winning_trades, losing_trades, total_profit_loss, average_win, average_loss, largest_win, largest_loss = result
            
            # Calculate win rate
            win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
            
            # Try to get actual account balance from MT5
            initial_balance = 10000  # Default fallback value
            try:
                # Import here to avoid circular imports
                from src.mt5_handler import MT5Handler
                mt5_handler = MT5Handler()
                if mt5_handler.initialize():
                    account_info = mt5_handler.get_account_info()
                    if account_info and 'balance' in account_info:
                        initial_balance = account_info['balance']
                    mt5_handler.shutdown()
            except Exception as e:
                logger.debug(f"Could not get actual balance from MT5: {str(e)}")
            
            # Calculate profit/loss percentage
            profit_loss_percentage = (total_profit_loss / initial_balance) * 100 if initial_balance > 0 else 0
            
            # Calculate drawdown (this is a simplified calculation)
            # In a real system, you would track the running balance and calculate max drawdown
            drawdown = 0
            if losing_trades > 0:
                cursor.execute(
                    """
                    SELECT 
                        open_time,
                        close_time,
                        profit_loss
                    FROM trades
                    WHERE status = 'closed' AND close_time >= ?
                    ORDER BY open_time ASC
                    """,
                    (start_date,)
                )
                
                trades_data = cursor.fetchall()
                
                # Calculate running balance and find max drawdown
                running_balance = initial_balance
                peak_balance = initial_balance
                max_drawdown = 0
                
                for _, _, profit_loss in trades_data:
                    running_balance += profit_loss
                    if running_balance > peak_balance:
                        peak_balance = running_balance
                    
                    current_drawdown = (peak_balance - running_balance) / peak_balance if peak_balance > 0 else 0
                    max_drawdown = max(max_drawdown, current_drawdown)
                
                drawdown = max_drawdown * 100  # Convert to percentage
            
            # Calculate Sharpe ratio (simplified)
            # In a real system, you would calculate this using daily returns
            sharpe_ratio = 0
            if total_trades > 0:
                # Get daily returns
                cursor.execute(
                    """
                    SELECT 
                        SUBSTR(close_time, 1, 10) as trade_date,
                        SUM(profit_loss) as daily_pnl
                    FROM trades
                    WHERE status = 'closed' AND close_time >= ?
                    GROUP BY SUBSTR(close_time, 1, 10)
                    """,
                    (start_date,)
                )
                
                daily_returns = [row[1] / initial_balance for row in cursor.fetchall()]
                
                if daily_returns:
                    import numpy as np
                    returns_array = np.array(daily_returns)
                    mean_return = np.mean(returns_array)
                    std_return = np.std(returns_array)
                    
                    # Annualized Sharpe ratio (assuming 252 trading days)
                    if std_return > 0:
                        sharpe_ratio = (mean_return / std_return) * np.sqrt(252)
            
            # Compile metrics
            metrics = {
                'date': now.strftime('%Y-%m-%d'),
                'timeframe': timeframe,
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'profit_loss': total_profit_loss,
                'profit_loss_percentage': profit_loss_percentage,
                'average_win': average_win or 0,
                'average_loss': average_loss or 0,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'drawdown': drawdown,
                'sharpe_ratio': sharpe_ratio
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            # Return default metrics
            return {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'timeframe': timeframe,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'profit_loss': 0.0,
                'profit_loss_percentage': 0.0,
                'average_win': 0.0,
                'average_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
    
    def get_closed_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get closed trades from the database.
        
        Args:
            limit: Maximum number of trades to return
            
        Returns:
            List of dictionaries containing closed trade data
        """
        try:
            if not self.conn:
                self.initialize_database()
            
            with self.lock:  # Use lock for thread safety
                cursor = self.conn.cursor()
                cursor.execute(
                    "SELECT * FROM trades WHERE status = 'closed' ORDER BY close_time DESC LIMIT ?",
                    (limit,)
                )
                
                # Get column names
                columns = [desc[0] for desc in cursor.description]
                
                # Convert rows to dictionaries
                trades = []
                for row in cursor.fetchall():
                    trade_dict = dict(zip(columns, row))
                    trades.append(trade_dict)
                
                return trades
        except Exception as e:
            logger.error(f"Error getting closed trades: {str(e)}")
            return []

# Singleton instance for easy access
db = TradingDatabase()

# For testing
if __name__ == "__main__":
    # Initialize the database
    test_db = TradingDatabase()
    
    # Test inserting a signal
    signal = {
        'timestamp': datetime.now().isoformat(),
        'symbol': 'EURUSD',
        'timeframe': 'H4',
        'strategy': 'sh_bms_rto',
        'direction': 'BUY',
        'entry_price': 1.10250,
        'stop_loss': 1.09750,
        'take_profit': 1.11250,
        'current_price': 1.10200,
        'confidence': 0.85,
        'description': 'SH+BMS+RTO Long: Stop Hunt + Bullish BMS + RTO to Bullish Order Block',
        'position_size': 0.1,
        'risk_amount': 50.0,
        'market_condition': 'trending',
        'volatility_state': 'low',
        'session': 'london',
        'htf_bias': 'bullish'
    }
    
    signal_id = test_db.insert_signal(signal)
    print(f"Inserted signal ID: {signal_id}")
    
    # Test inserting a trade
    trade = {
        'ticket': '123456',
        'symbol': 'EURUSD',
        'direction': 'BUY',
        'entry_price': 1.10250,
        'current_price': 1.10300,
        'stop_loss': 1.09750,
        'take_profit': 1.11250,
        'position_size': 0.1,
        'open_time': datetime.now().isoformat(),
        'status': 'open'
    }
    
    trade_id = test_db.insert_trade(trade, signal_id)
    print(f"Inserted trade ID: {trade_id}")
    
    # Test getting active signals
    active_signals = test_db.get_active_signals()
    print(f"Active signals: {len(active_signals)}")
    
    # Test getting active trades
    active_trades = test_db.get_active_trades()
    print(f"Active trades: {len(active_trades)}")
    
    # Test calculating performance metrics
    metrics = test_db.calculate_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    # Close the database
    test_db.close() 