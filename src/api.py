from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import json
from pathlib import Path
from loguru import logger
import os
import re
import MetaTrader5 as mt5
from datetime import datetime, time, UTC
import numpy as np

from config.config import (
    TRADING_CONFIG,
    SESSION_CONFIG,
    SIGNAL_THRESHOLDS,
    BASE_DIR
)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConfigUpdate(BaseModel):
    trading_config: Dict
    session_config: Dict
    signal_thresholds: Dict

def update_config_section(content: str, section_name: str, new_config: Dict) -> str:
    """Update a specific section in the config file using regex pattern matching."""
    try:
        # Create pattern that matches the entire dictionary definition
        pattern = f"{section_name} = {{[^}}]*}}"
        
        # Format the new config as a proper Python dictionary string
        new_section = f"{section_name} = {{\n"
        for key, value in new_config.items():
            if isinstance(value, str):
                new_section += f'    "{key}": "{value}",\n'
            elif isinstance(value, (list, tuple)):
                new_section += f'    "{key}": {value},\n'
            else:
                new_section += f'    "{key}": {value},\n'
        new_section += "}"
        
        # Replace the old section with the new one
        updated_content = re.sub(pattern, new_section, content, flags=re.DOTALL)
        return updated_content
    except Exception as e:
        logger.error(f"Error updating config section {section_name}: {str(e)}")
        raise

def update_config_file(config_updates: Dict) -> None:
    """Update the configuration file with new values."""
    config_file = BASE_DIR / "config" / "config.py"
    
    if not config_file.exists():
        raise HTTPException(status_code=404, detail="Configuration file not found")
    
    try:
        # Read the current config file
        with open(config_file, "r") as f:
            config_content = f.read()
        
        # Update each section if present in updates
        if "trading_config" in config_updates:
            new_trading_config = {**TRADING_CONFIG, **config_updates["trading_config"]}
            config_content = update_config_section(
                config_content, 
                "TRADING_CONFIG",
                new_trading_config
            )
        
        if "session_config" in config_updates:
            new_session_config = {**SESSION_CONFIG, **config_updates["session_config"]}
            config_content = update_config_section(
                config_content,
                "SESSION_CONFIG",
                new_session_config
            )
        
        if "signal_thresholds" in config_updates:
            new_signal_thresholds = {**SIGNAL_THRESHOLDS, **config_updates["signal_thresholds"]}
            config_content = update_config_section(
                config_content,
                "SIGNAL_THRESHOLDS",
                new_signal_thresholds
            )
        
        # Write the updated config back to file
        with open(config_file, "w") as f:
            f.write(config_content)
            
        logger.info("Configuration updated successfully")
        
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

@app.post("/api/update-config")
async def update_config(config: ConfigUpdate):
    """Update the trading bot configuration."""
    try:
        update_config_file(config.dict())
        
        return {
            "success": True,
            "message": "Configuration updated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error in update_config endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
async def get_config():
    """Get the current trading bot configuration."""
    try:
        return {
            "trading_config": TRADING_CONFIG,
            "session_config": SESSION_CONFIG,
            "signal_thresholds": SIGNAL_THRESHOLDS
        }
    except Exception as e:
        logger.error(f"Error in get_config endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/active-trades")
async def get_active_trades():
    """Get the current active trades."""
    try:
        trades_file = Path(BASE_DIR) / "data" / "active_trades.json"
        
        if not trades_file.exists():
            logger.warning(f"Active trades file not found: {trades_file}")
            return {"active_trades": []}
        
        with open(trades_file, "r") as f:
            trades_data = json.load(f)
        
        return {
            "active_trades": trades_data
        }
        
    except Exception as e:
        logger.error(f"Error in get_active_trades endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trading-status")
async def get_trading_status():
    """Get current trading status and performance metrics."""
    try:
        # Initialize MT5 if not already initialized
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            return {
                "error": "MT5 not initialized",
                "trading_status": "Disabled",
                "performance": {
                    "total_return": 0,
                    "win_rate": 0,
                    "max_drawdown": 0,
                    "sharpe_ratio": 0,
                    "risk_reward_ratio": 0,
                    "profit_factor": 0,
                    "average_win": 0,
                    "average_loss": 0,
                    "largest_win": 0,
                    "largest_loss": 0,
                    "consecutive_wins": 0,
                    "consecutive_losses": 0
                },
                "account_info": {
                    "balance": 0,
                    "equity": 0,
                    "profit": 0,
                    "margin": 0,
                    "margin_free": 0,
                    "margin_level": 0,
                    "currency": "USD"
                },
                "active_trades": [],
                "equity_curve": []
            }

        # Get account info
        account_info = mt5.account_info()
        if not account_info:
            raise Exception("Failed to get account info")

        # Get all open positions
        positions = mt5.positions_get()
        active_trades = []
        
        if positions:
            for pos in positions:
                active_trades.append({
                    "ticket": pos.ticket,
                    "symbol": pos.symbol,
                    "type": "BUY" if pos.type == mt5.ORDER_TYPE_BUY else "SELL",
                    "volume": pos.volume,
                    "entry_price": pos.price_open,
                    "current_price": pos.price_current,
                    "sl": pos.sl,
                    "tp": pos.tp,
                    "profit": pos.profit,
                    "swap": pos.swap,
                    "time": pos.time
                })

        # Get profit history for performance calculations
        history_file = Path(BASE_DIR) / "data" / "profit_history.json"
        history_data = []
        if history_file.exists():
            with open(history_file, 'r') as f:
                history_data = json.load(f)

        # Calculate performance metrics
        total_trades = len(history_data)
        if total_trades > 0:
            # Calculate win rate
            winning_trades = len([trade for trade in history_data if trade['profit'] > 0])
            win_rate = (winning_trades / total_trades) * 100

            # Calculate profit metrics
            profits = [trade['profit'] for trade in history_data]
            winning_profits = [p for p in profits if p > 0]
            losing_profits = [p for p in profits if p < 0]

            # Average metrics
            average_win = sum(winning_profits) / len(winning_profits) if winning_profits else 0
            average_loss = abs(sum(losing_profits) / len(losing_profits)) if losing_profits else 0
            
            # Risk-reward ratio
            risk_reward_ratio = average_win / average_loss if average_loss != 0 else 0

            # Profit factor
            gross_profit = sum(winning_profits)
            gross_loss = abs(sum(losing_profits))
            profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0

            # Largest win/loss
            largest_win = max(winning_profits) if winning_profits else 0
            largest_loss = abs(min(losing_profits)) if losing_profits else 0

            # Calculate consecutive wins/losses
            consecutive_wins = 0
            consecutive_losses = 0
            current_streak = 0
            for profit in profits:
                if profit > 0:
                    if current_streak > 0:
                        current_streak += 1
                    else:
                        current_streak = 1
                    consecutive_wins = max(consecutive_wins, current_streak)
                else:
                    if current_streak < 0:
                        current_streak -= 1
                    else:
                        current_streak = -1
                    consecutive_losses = max(consecutive_losses, abs(current_streak))

            # Calculate max drawdown
            cumulative = np.array([trade['cumulative'] for trade in history_data])
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / peak * 100
            max_drawdown = np.max(drawdown)

            # Calculate Sharpe ratio (assuming risk-free rate of 0.02 or 2%)
            returns = np.diff(cumulative) / cumulative[:-1]
            excess_returns = returns - 0.02/252  # Daily risk-free rate
            sharpe_ratio = np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns) if len(returns) > 0 else 0

            # Generate equity curve data
            equity_curve = [
                {
                    "timestamp": entry["timestamp"],
                    "equity": entry["cumulative"],
                    "drawdown": float(drawdown[i])
                }
                for i, entry in enumerate(history_data)
            ]

        else:
            # Default values if no trades
            win_rate = 0
            max_drawdown = 0
            sharpe_ratio = 0
            risk_reward_ratio = 0
            profit_factor = 0
            average_win = 0
            average_loss = 0
            largest_win = 0
            largest_loss = 0
            consecutive_wins = 0
            consecutive_losses = 0
            equity_curve = []

        # Calculate total return
        total_return = ((account_info.equity - account_info.balance) / account_info.balance * 100) if account_info.balance > 0 else 0

        return {
            "trading_status": "Enabled" if mt5.terminal_info().connected else "Disabled",
            "performance": {
                "total_return": total_return,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "sharpe_ratio": sharpe_ratio,
                "risk_reward_ratio": risk_reward_ratio,
                "profit_factor": profit_factor,
                "average_win": average_win,
                "average_loss": average_loss,
                "largest_win": largest_win,
                "largest_loss": largest_loss,
                "consecutive_wins": consecutive_wins,
                "consecutive_losses": consecutive_losses
            },
            "account_info": {
                "balance": account_info.balance,
                "equity": account_info.equity,
                "profit": account_info.profit,
                "margin": account_info.margin,
                "margin_free": account_info.margin_free,
                "margin_level": account_info.margin_level,
                "currency": account_info.currency
            },
            "active_trades": active_trades,
            "equity_curve": equity_curve,
            "last_update": datetime.now(UTC).isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting trading status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/profit-history")
async def get_profit_history(timeframe: str = "24H"):
    """Get profit history with timeframe filtering."""
    try:
        logger.info(f"Fetching profit history for timeframe: {timeframe}")
        history_file = Path(BASE_DIR) / "data" / "profit_history.json"
        
        if not history_file.exists():
            logger.warning(f"History file not found: {history_file}")
            return {
                "data": [],
                "cumulative_profit": 0,
                "timeframe": timeframe
            }
        
        with open(history_file, "r") as f:
            history_data = json.load(f)
        
        if not history_data:
            logger.warning("Empty history data")
            return {
                "data": [],
                "cumulative_profit": 0,
                "timeframe": timeframe
            }
            
        # Sort data by timestamp
        history_data.sort(key=lambda x: x["timestamp"])
        
        # Use the latest timestamp as reference point instead of current time
        latest_time = max(datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00")) for entry in history_data)
        logger.debug(f"Latest timestamp in data: {latest_time}")
        
        # Filter based on timeframe relative to the latest timestamp
        filtered_data = []
        for entry in history_data:
            try:
                # Parse timestamp and ensure it's UTC
                entry_time = datetime.fromisoformat(entry["timestamp"].replace("Z", "+00:00"))
                if entry_time.tzinfo is None:
                    entry_time = entry_time.replace(tzinfo=UTC)
                
                # Calculate time difference from latest timestamp
                time_diff = (latest_time - entry_time).total_seconds()
                
                # Apply timeframe filter
                if timeframe == "24H" and time_diff <= 86400:
                    filtered_data.append(entry)
                elif timeframe == "7D" and time_diff <= 604800:
                    filtered_data.append(entry)
                elif timeframe == "30D" and time_diff <= 2592000:
                    filtered_data.append(entry)
                elif timeframe == "ALL":
                    filtered_data.append(entry)
                    
                logger.debug(f"Processed entry: {entry['timestamp']} (diff: {time_diff}s)")
            except Exception as e:
                logger.error(f"Error processing entry {entry}: {str(e)}")
                continue
        
        logger.info(f"Filtered {len(filtered_data)} entries for timeframe {timeframe}")
        
        # Ensure cumulative values are properly calculated
        running_total = 0
        formatted_data = []
        for entry in filtered_data:
            try:
                running_total += entry["profit"]
                formatted_entry = {
                    "timestamp": entry["timestamp"],
                    "profit": entry["profit"],
                    "trade_type": entry.get("trade_type", None),
                    "cumulative": running_total
                }
                formatted_data.append(formatted_entry)
                logger.debug(f"Formatted entry: {formatted_entry}")
            except Exception as e:
                logger.error(f"Error formatting entry {entry}: {str(e)}")
                continue
        
        response_data = {
            "data": formatted_data,
            "cumulative_profit": running_total,
            "timeframe": timeframe
        }
        
        logger.info(f"Returning {len(formatted_data)} entries with cumulative profit {running_total}")
        return response_data
        
    except Exception as e:
        logger.error(f"Error getting profit history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trading-data")
async def get_trading_data():
    """Get trading data for the dashboard."""
    try:
        # Initialize MT5 if not already initialized
        if not mt5.initialize():
            raise HTTPException(status_code=500, detail="MT5 not initialized")
            
        # Get account info
        account_info = mt5.account_info()
        if not account_info:
            raise HTTPException(status_code=500, detail="Failed to get account info")
        
        # Get active trades
        positions = mt5.positions_get()
        active_trades = []
        
        if positions:
            for pos in positions:
                profit_percent = (pos.profit / account_info.balance) * 100 if account_info.balance > 0 else 0
                active_trades.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
                    'volume': pos.volume,
                    'entry_price': pos.price_open,
                    'current_price': pos.price_current,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'profit': pos.profit,
                    'profit_percent': profit_percent,
                    'swap': pos.swap,
                    'time': pos.time
                })
        
        # Get profit history for calculations
        history_file = Path(BASE_DIR) / "data" / "profit_history.json"
        history_data = []
        if history_file.exists():
            with open(history_file, 'r') as f:
                history_data = json.load(f)
        
        # Calculate win rate
        total_trades = len(history_data)
        winning_trades = len([trade for trade in history_data if trade['profit'] > 0]) if history_data else 0
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate total profit
        total_profit = sum(trade['profit'] for trade in history_data) if history_data else 0
        
        return {
            'mt5_account': {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'profit': account_info.profit,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'currency': account_info.currency
            },
            'total_profit': total_profit,
            'daily_profit': sum(
                trade['profit'] 
                for trade in history_data 
                if datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00')).date() == datetime.now().date()
            ) if history_data else 0,
            'winRate': round(win_rate, 2),
            'totalTrades': total_trades,
            'active_trades': active_trades,
            'last_update': datetime.now(UTC).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting trading data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 