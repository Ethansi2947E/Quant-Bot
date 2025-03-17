from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from datetime import datetime, timedelta
import uvicorn 
from loguru import logger
import os
import subprocess
import sys
from pathlib import Path
import traceback
import time
import random
import json

# Use TYPE_CHECKING for type hints to avoid runtime imports
if TYPE_CHECKING:
    from src.trading_bot import TradingBot

# Import components that don't create circular dependencies
from src.mt5_handler import MT5Handler
from src.database import db

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {str(e)}")
                # Remove failed connection
                self.active_connections.remove(connection)

class DashboardAPI:
    def __init__(self, trading_bot: Optional['TradingBot'] = None):
        self.app = FastAPI(title="Trading Bot Dashboard API")
        self.trading_bot = trading_bot
        
        # If no trading bot provided, create a new instance
        if self.trading_bot is None:
            # Import here to avoid circular dependencies
            from src.trading_bot import TradingBot
            self.trading_bot = TradingBot()
            self._owns_trading_bot = True
        else:
            self._owns_trading_bot = False
            
        self.manager = ConnectionManager()
        self.last_broadcast = datetime.now()
        self.broadcast_interval = 1.0  # seconds, use float to avoid type issues
        self.frontend_process = None
        
        # Setup CORS for development
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Allow all origins in development
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Register routes
        self.register_routes()
        
    def register_routes(self):
        """Register API routes for the dashboard"""
        # WebSocket endpoint
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.manager.connect(websocket)
            try:
                # Start a background task to periodically send updates to this client
                asyncio.create_task(self.broadcast_data(websocket))
                
                while True:
                    # Wait for messages from the client
                    data = await websocket.receive_text()
                    try:
                        # Parse the incoming message
                        message = json.loads(data)
                        command = message.get("command", "")
                        
                        # Handle the various commands
                        if command == "get_overview":
                            # Send back account overview
                            overview = self.get_account_overview()
                            await websocket.send_json({
                                "type": "overview",
                                "data": overview
                            })
                            
                        elif command == "get_active_trades":
                            # Send back active trades
                            active_trades = self.get_active_trades()
                            await websocket.send_json({
                                "type": "active_trades",
                                "data": active_trades
                            })
                            
                        elif command == "get_performance":
                            # Send back performance metrics
                            timeframe = message.get("timeframe", "1D")
                            performance_data = self.get_performance_metrics(timeframe=timeframe)
                            await websocket.send_json({
                                "type": "performance",
                                "data": performance_data
                            })
                            
                        elif command == "get_recent_activity":
                            # Send back recent activity
                            recent_activity = self.get_recent_activity()
                            await websocket.send_json({
                                "type": "recent_activity",
                                "data": recent_activity
                            })
                            
                        elif command == "get_trade_history":
                            # Send back trade history with detailed profit/loss
                            limit = message.get("limit", 100)
                            offset = message.get("offset", 0)
                            include_active = message.get("include_active", True)
                            
                            trade_history = self.get_detailed_trade_history(
                                limit=limit,
                                offset=offset,
                                include_active=include_active
                            )
                            await websocket.send_json({
                                "type": "trade_history",
                                "data": trade_history
                            })
                            
                        elif command == "ping":
                            # Simple ping-pong for connection testing
                            await websocket.send_json({
                                "type": "pong",
                                "timestamp": datetime.now().isoformat()
                            })
                            
                        else:
                            # Unknown command
                            await websocket.send_json({
                                "type": "error",
                                "message": f"Unknown command: {command}"
                            })
                            
                    except json.JSONDecodeError:
                        logger.warning(f"Received invalid JSON: {data}")
                        # Send error message
                        await websocket.send_json({
                            "type": "error",
                            "message": "Invalid JSON format"
                        })
                        
                    except Exception as e:
                        logger.error(f"Error processing WebSocket message: {str(e)}")
                        # Send error message
                        await websocket.send_json({
                            "type": "error",
                            "message": "Internal server error"
                        })
                        
                    # Wait a small amount to not consume CPU
                    await asyncio.sleep(0.1)
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
                self.manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
                if websocket in self.manager.active_connections:
                    self.manager.disconnect(websocket)
        
        # Account overview endpoint
        @self.app.get("/api/dashboard/overview")
        async def get_overview():
            return self.get_account_overview()
        
        # Active trades endpoint
        @self.app.get("/api/dashboard/active-trades")
        async def get_active_trades():
            return self.get_active_trades()
        
        # Performance metrics endpoint
        @self.app.get("/api/dashboard/performance")
        async def get_performance(timeframe: str = "1D"):
            # Get the timeframe from the query parameter with a default of 1D
            return self.get_performance_metrics(timeframe)
            
        # Recent activity endpoint
        @self.app.get("/api/dashboard/recent-activity")
        async def get_recent_activity():
            return self.get_recent_activity()
        
        # NEW: Trade history endpoint with profit/loss details
        @self.app.get("/api/dashboard/trade-history")
        async def get_trade_history(limit: int = 100, offset: int = 0, include_active: bool = True):
            return self.get_detailed_trade_history(limit, offset, include_active)
            
        # Trading signals endpoint
        @self.app.get("/api/dashboard/signals")
        async def get_signals():
            return self.get_trading_signals()
            
        # Account history endpoint
        @self.app.get("/api/dashboard/account-history")
        async def get_account_history():
            return self.get_account_history_data()
            
        # Startup frontend endpoint
        @self.app.get("/api/dashboard/start-frontend")
        async def start_frontend():
            success = self.start_frontend()
            return {"success": success, "message": "Frontend started" if success else "Failed to start frontend"}
        
        # Health check endpoint to verify API is running
        @self.app.get("/api/health")
        async def health_check():
            mt5_connected = self.trading_bot.mt5_handler.connected
            
            # Force reconnect if disconnected
            if not mt5_connected:
                mt5_connected = self.trading_bot.mt5_handler.initialize()
                
            return {
                "status": "online",
                "timestamp": datetime.now().isoformat(),
                "mt5_connected": mt5_connected,
                "active_websockets": len(self.manager.active_connections)
            }
            
    async def broadcast_data(self, websocket: WebSocket):
        """Broadcast real-time data to the connected client"""
        try:
            # Check if the websocket is still connected
            if websocket not in self.manager.active_connections:
                logger.debug("Skipping broadcast to disconnected websocket")
                return
                
            # Send account overview data
            account_data = self.get_account_overview()
            await websocket.send_json({
                "type": "account_overview",
                "data": account_data
            })
            
            # Send performance metrics for charts
            performance_data = self.get_performance_metrics(timeframe="1D")
            await websocket.send_json({
                "type": "performance_metrics",
                "data": performance_data
            })
            
            # Send recent activity data
            recent_activity = self.get_recent_activity()
            await websocket.send_json({
                "type": "recent_activity",
                "data": recent_activity
            })
            
            # Send trade history data (limit to 10 for regular broadcasts to keep it lightweight)
            trade_history = self.get_detailed_trade_history(limit=10, offset=0, include_active=True)
            await websocket.send_json({
                "type": "trade_history",
                "data": trade_history
            })
            
            # Send active trades as a separate broadcast
            active_trades = self.get_active_trades()
            await websocket.send_json({
                    "type": "active_trades",
                    "data": active_trades
                })
        except WebSocketDisconnect:
            # Handle proper websocket disconnection
            logger.info("WebSocket disconnected during broadcast")
            if websocket in self.manager.active_connections:
                self.manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"Error broadcasting data: {str(e)}")
            # Remove the problematic connection
            if websocket in self.manager.active_connections:
                self.manager.disconnect(websocket)
            
    def get_account_overview(self) -> Dict[str, Any]:
        """Get account overview data for the dashboard"""
        try:
            # Force reconnect to MT5 if needed to ensure fresh data
            if not self.trading_bot.mt5_handler.connected:
                logger.warning("MT5 connection lost, attempting to reconnect for dashboard data")
                self.trading_bot.mt5_handler.initialize()
                
            # Get account info directly from MT5 with retry
            max_retries = 3
            account_info = None
            
            for attempt in range(max_retries):
                account_info = self.trading_bot.mt5_handler.get_account_info()
                if account_info and account_info.get("free_margin", 0) > 0:
                    break
                    
                if attempt < max_retries - 1:
                    logger.warning(f"Account data fetch attempt {attempt+1} failed or returned zero free margin. Retrying...")
                    time.sleep(0.5)  # Brief delay before retry
            
            # Fallback if still no valid data after retries
            if not account_info or len(account_info) == 0:
                logger.error("Failed to get valid account data after multiple attempts")
                account_info = {
                    "balance": 0,
                    "equity": 0, 
                    "free_margin": 0
                }
            
            # Get maximum drawdown with retry logic
            max_drawdown = 0.0
            try:
                weekly_metrics = db.get_performance_metrics(timeframe='1W', limit=1)
                if weekly_metrics and len(weekly_metrics) > 0:
                    max_drawdown = weekly_metrics[0].get('drawdown', 0.0)
                else:
                    # Calculate metrics if none exist
                    metrics = db.calculate_performance_metrics(timeframe='1W')
                    max_drawdown = metrics.get('drawdown', 0.0)
            except Exception as e:
                logger.error(f"Error getting max drawdown: {str(e)}")
            
            # Calculate risk level based on drawdown
            risk_level = "Low"
            if max_drawdown > 10:
                risk_level = "High"
            elif max_drawdown > 5:
                risk_level = "Medium"
                
            # Get daily PnL (more reliable calculation)
            balance = account_info.get("balance", 0)
            equity = account_info.get("equity", 0)
            daily_pnl = equity - balance
            
            # Get open positions count
            open_positions = len(self.trading_bot.mt5_handler.get_open_positions())
            
            # Get total trades count (active + closed)
            total_trades = 0
            try:
                # Get active trades from database
                active_trades = db.get_active_trades()
                # Get closed trades from database (get all closed trades)
                closed_trades = db.get_closed_trades(limit=1000)  # Use a high limit to get all trades
                
                # Total trades is the sum of active and closed trades
                total_trades = len(active_trades) + len(closed_trades)
                
                logger.debug(f"Total trades count: {total_trades} (Active: {len(active_trades)}, Closed: {len(closed_trades)})")
            except Exception as e:
                logger.error(f"Error getting total trades count: {str(e)}")
            
            # Log the values we're returning
            logger.debug(f"Dashboard data: Balance={balance}, Equity={equity}, Free Margin={account_info.get('free_margin', 0)}, Drawdown={max_drawdown}, Total Trades={total_trades}")
            
            return {
                "balance": balance,
                "equity": equity,
                "freeMargin": account_info.get("free_margin", 0),
                "maxDrawdown": max_drawdown,
                "riskLevel": risk_level,
                "dailyPnL": daily_pnl,
                "openPositions": open_positions,
                "totalTrades": total_trades  # Add the total trades count
            }
        except Exception as e:
            logger.error(f"Error getting account overview: {str(e)}")
            logger.error(traceback.format_exc())
            # Return fallback data
            return {
                "balance": 0,
                "equity": 0,
                "freeMargin": 0,
                "maxDrawdown": 0,
                "riskLevel": "Unknown",
                "dailyPnL": 0,
                "openPositions": 0,
                "totalTrades": 0
            }
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get active trades for the dashboard"""
        # Get trades from both database and MT5
        db_trades = db.get_active_trades()
        mt5_positions = self.trading_bot.mt5_handler.get_open_positions()
        
        formatted_trades = []
        
        # Process database trades
        for trade in db_trades:
            # Calculate duration
            current_time = datetime.now()
            
            # Handle different time formats
            open_time_str = trade.get("open_time")
            if open_time_str is None:
                duration_str = "0h 0m"
            else:
                try:
                    open_time = datetime.fromisoformat(open_time_str.replace('Z', '+00:00'))
                    duration = current_time - open_time
                    hours, remainder = divmod(duration.total_seconds(), 3600)
                    minutes, _ = divmod(remainder, 60)
                    duration_str = f"{int(hours)}h {int(minutes)}m"
                except Exception:
                    duration_str = "N/A"
            
            # Calculate profit/loss percentage
            entry_price = trade.get("entry_price", 0)
            current_price = trade.get("current_price", entry_price)
            profit_loss = trade.get("profit_loss", 0)
            
            # Calculate percentage - this is simplified and should be more sophisticated in practice
            if entry_price > 0:
                if trade.get("direction", "").upper() == "BUY":
                    profit_loss_percentage = ((current_price - entry_price) / entry_price) * 100
                else:
                    profit_loss_percentage = ((entry_price - current_price) / entry_price) * 100
            else:
                profit_loss_percentage = 0
            
            formatted_trades.append({
                "id": str(trade.get("id", "")),
                "ticket": trade.get("ticket", ""),
                "symbol": trade.get("symbol", ""),
                "direction": trade.get("direction", "BUY").upper(),
                "type": trade.get("direction", "BUY").upper(),
                "entryPrice": entry_price,
                "currentPrice": current_price,
                "stopLoss": trade.get("stop_loss", 0),
                "takeProfit": trade.get("take_profit", 0),
                "duration": duration_str,
                "status": "profit" if profit_loss > 0 else "loss",
                "profitLoss": profit_loss,
                "profitLossPercentage": profit_loss_percentage,
                "size": float(trade.get("volume", 0.0)),
                "strategy": trade.get("strategy", "Unknown"),
                "openTime": trade.get("open_time", ""),
                "closeTime": "Active",
                "isActive": trade.get("is_active", True)
            })
        
        # Process MT5 positions (if not already in database)
        db_tickets = {trade.get("ticket", "") for trade in db_trades}
        
        for position in mt5_positions:
            ticket = str(position.get("ticket", ""))
            if ticket and ticket not in db_tickets:
                # Calculate duration
                current_time = datetime.now()
                
                # Handle different time formats
                open_time = position.get("time")
                if open_time is None:
                    duration_str = "0h 0m"
                else:
                    try:
                        if isinstance(open_time, datetime):
                            duration = current_time - open_time
                            hours, remainder = divmod(duration.total_seconds(), 3600)
                            minutes, _ = divmod(remainder, 60)
                            duration_str = f"{int(hours)}h {int(minutes)}m"
                        else:
                            duration_str = "N/A"
                    except Exception:
                        duration_str = "N/A"
                
                # Handle profit/loss calculation
                profit = position.get("profit", 0)
                volume = position.get("volume", 0)
                
                # Calculate percentage (simplified)
                entry_price = position.get("price_open", 0)
                current_price = position.get("price_current", entry_price)
                profit_loss_pct = 0
                
                if entry_price > 0 and volume > 0:
                    direction = "buy" if position.get("type") == 0 else "sell"
                    if direction == "buy":
                        profit_loss_pct = ((current_price - entry_price) / entry_price) * 100
                    else:
                        profit_loss_pct = ((entry_price - current_price) / entry_price) * 100
                
                formatted_trades.append({
                    "id": ticket,
                    "ticket": ticket,
                    "symbol": position.get("symbol", ""),
                    "direction": direction.upper(),
                    "type": direction.upper(),
                    "entryPrice": entry_price,
                    "currentPrice": current_price,
                    "stopLoss": position.get("sl", 0),
                    "takeProfit": position.get("tp", 0),
                    "duration": duration_str,
                    "status": "profit" if profit > 0 else "loss",
                    "profitLoss": profit,
                    "profitLossPercentage": profit_loss_pct,
                    "size": float(volume),
                    "strategy": "Unknown",
                    "openTime": "N/A",
                    "closeTime": "Active",
                    "isActive": True
                })
        
        return formatted_trades
    
    def get_trading_signals(self) -> List[Dict[str, Any]]:
        """Get trading signals from the database for the dashboard"""
        try:
            # Get active signals from the database
            db_signals = db.get_active_signals()
            
            # Format signals for the dashboard
            formatted_signals = []
            for signal in db_signals:
                # Convert confidence from 0-1 scale to descriptive terms
                confidence_score = signal.get("confidence", 0)
                if isinstance(confidence_score, str):
                    try:
                        confidence_score = float(confidence_score)
                    except ValueError:
                        confidence_score = 0
                
                confidence = "low"
                if confidence_score >= 0.7:
                    confidence = "high"
                elif confidence_score >= 0.4:
                    confidence = "medium"
                
                # Format the timestamp
                timestamp = signal.get("timestamp", datetime.now().isoformat())
                if not isinstance(timestamp, str):
                    timestamp = timestamp.isoformat()
                
                formatted_signals.append({
                    "id": str(signal.get("id", "")),
                    "timestamp": timestamp,
                    "symbol": signal.get("symbol", ""),
                    "type": signal.get("direction", "").lower(),
                    "price": signal.get("entry_price", 0),
                    "confidence": confidence,
                    "timeframe": signal.get("timeframe", ""),
                    "strategy": signal.get("strategy", ""),
                    "explanation": signal.get("description", ""),
                    "stop_loss": signal.get("stop_loss", 0),
                    "take_profit": signal.get("take_profit", 0),
                    "market_condition": signal.get("market_condition", ""),
                    "session": signal.get("session", ""),
                })
            
            return formatted_signals
        except Exception as e:
            logger.error(f"Error getting trading signals: {str(e)}")
            return []
    
    def get_performance_metrics(self, timeframe: str) -> Dict[str, Any]:
        """Get performance metrics for charts"""
        try:
            # Get performance data for each timeframe (daily, weekly, monthly)
            daily_metrics = db.get_performance_metrics(timeframe='1D', limit=7)
            weekly_metrics = db.get_performance_metrics(timeframe='1W', limit=7) 
            monthly_metrics = db.get_performance_metrics(timeframe='1M', limit=7)
            
            # If no metrics are available, calculate them
            if not daily_metrics:
                logger.info("No daily metrics found, calculating new metrics")
                daily_data = db.calculate_performance_metrics(timeframe='1D')
                db.update_or_insert_performance_metrics(daily_data)
                daily_metrics = [daily_data]
                
            if not weekly_metrics:
                logger.info("No weekly metrics found, calculating new metrics")
                weekly_data = db.calculate_performance_metrics(timeframe='1W')
                db.update_or_insert_performance_metrics(weekly_data)
                weekly_metrics = [weekly_data]
                
            if not monthly_metrics:
                logger.info("No monthly metrics found, calculating new metrics")
                monthly_data = db.calculate_performance_metrics(timeframe='1M')
                db.update_or_insert_performance_metrics(monthly_data)
                monthly_metrics = [monthly_data]
            
            # Get actual balance history from MT5 (if available)
            balance_history = []
            try:
                # For real implementation, try to get this from MT5 or database history
                history = self.trading_bot.mt5_handler.get_account_history(7)  # Last 7 days
                if history and len(history) > 0:
                    balance_history = history
            except Exception as e:
                logger.error(f"Failed to get balance history from MT5: {str(e)}")
            
            # Format chart data for each timeframe
            daily_chart_data = self._format_chart_data(daily_metrics, balance_history, '1D')
            weekly_chart_data = self._format_chart_data(weekly_metrics, balance_history, '1W')
            monthly_chart_data = self._format_chart_data(monthly_metrics, balance_history, '1M')
            
            # Get latest metrics for summary data
            latest_metrics = daily_metrics[0] if daily_metrics else {}
            
            # Get portfolio value from MT5 (current equity)
            portfolio_value = 0
            portfolio_growth = 0
            try:
                account_info = self.trading_bot.mt5_handler.get_account_info()
                if account_info:
                    portfolio_value = account_info.get("equity", 0)
                    # Calculate growth based on starting balance if available
                    starting_balance = account_info.get("balance_start", account_info.get("balance", 0))
                    if starting_balance > 0:
                        portfolio_growth = ((portfolio_value - starting_balance) / starting_balance) * 100
            except Exception as e:
                logger.error(f"Failed to get account info for portfolio value: {str(e)}")
            
            return {
                "totalTrades": latest_metrics.get("total_trades", 0),
                "winRate": latest_metrics.get("win_rate", 0),
                "profitLoss": latest_metrics.get("profit_loss", 0),
                "profitLossPercentage": latest_metrics.get("profit_loss_percentage", 0),
                "portfolioValue": portfolio_value,
                "portfolioGrowth": portfolio_growth,
                "timeframe": timeframe,
                "chartData": {
                    "daily": daily_chart_data,
                    "weekly": weekly_chart_data,
                    "monthly": monthly_chart_data
                }
            }
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            logger.error(traceback.format_exc())
            # Return mock data as fallback with proper structure
            return {
                "totalTrades": 0,
                "winRate": 0,
                "profitLoss": 0,
                "profitLossPercentage": 0,
                "portfolioValue": 0,
                "portfolioGrowth": 0,
                "timeframe": timeframe,
                "chartData": {
                    "daily": self._get_mock_chart_data('1D'),
                    "weekly": self._get_mock_chart_data('1W'),
                    "monthly": self._get_mock_chart_data('1M')
                }
            }
            
    def _format_chart_data(self, metrics_data, balance_history, timeframe):
        """Format metrics data for charts with the right structure"""
        # Ensure we have some data to work with
        if not metrics_data or len(metrics_data) == 0:
            return self._get_mock_chart_data(timeframe)
            
        try:
            chart_data = []
        
            # Track min/max values for chart scaling
            min_value = float("inf")
            max_value = float("-inf")
            
            # Try to use the actual balance history first
            if balance_history and len(balance_history) > 0:
                # Sort balance history by date
                balance_history = sorted(
                    balance_history, key=lambda x: x.get("date", "")
                )
                
                # Format data points
                for entry in balance_history:
                    date_str = entry.get("date", "")
                    if not date_str:
                        continue
                        
                    # Format date based on timeframe
                    if isinstance(date_str, str):
                        try:
                            date_obj = datetime.fromisoformat(
                                date_str.replace("Z", "+00:00")
                            )
                        except Exception:
                            try:
                                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                            except Exception:
                                continue
                    else:
                        date_obj = date_str
                        
                    # Format date string based on timeframe
                    if timeframe == "1D":
                        formatted_date = date_obj.strftime("%m/%d")
                    elif timeframe == "1W":
                        formatted_date = f"Week {date_obj.isocalendar()[1]}"
                    else:
                        formatted_date = date_obj.strftime("%b %Y")
                        
                    # Get value and update min/max
                    value = entry.get("balance", 0)
                    min_value = min(min_value, value)
                    max_value = max(max_value, value)
                    
                    chart_data.append({
                        "date": formatted_date,
                        "value": value
                    })
            else:
                # Fall back to metrics data if balance history is not available
                for metric in metrics_data:
                    # Get the date
                    date_str = metric.get("timeframe_start", "")
                    if not date_str:
                        continue
                        
                    # Format date based on timeframe
                    if isinstance(date_str, str):
                        try:
                            date_obj = datetime.fromisoformat(
                                date_str.replace("Z", "+00:00")
                            )
                        except Exception:
                            try:
                                date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                            except Exception:
                                continue
                    else:
                        date_obj = date_str
                        
                    # Format date string based on timeframe
                    if timeframe == "1D":
                        formatted_date = date_obj.strftime("%m/%d")
                    elif timeframe == "1W":
                        formatted_date = f"Week {date_obj.isocalendar()[1]}"
                    else:
                        formatted_date = date_obj.strftime("%b %Y")
                        
                    # Use profit_loss for the value
                    value = metric.get("profit_loss", 0)
                    min_value = min(min_value, value)
                    max_value = max(max_value, value)
                    
                    chart_data.append({
                        "date": formatted_date,
                        "value": value
                    })
                    
            # Make sure we have some padding in the chart
            if min_value == max_value:
                min_value -= 1
                max_value += 1
                
            # Sort chart data by date
            chart_data = sorted(chart_data, key=lambda x: x["date"])
            
            # Ensure we have at least 7 data points for a good chart
            if len(chart_data) < 7:
                # Fill in missing data points
                existing_dates = [entry["date"] for entry in chart_data]
                
                # Generate placeholder dates
                today = datetime.now()
                for i in range(7 - len(chart_data)):
                    date_offset = today - timedelta(days=i + 1)
                    
                    if timeframe == "1D":
                        formatted_date = date_offset.strftime("%m/%d")
                    elif timeframe == "1W":
                        formatted_date = f"Week {date_offset.isocalendar()[1]}"
                    else:
                        formatted_date = date_offset.strftime("%b %Y")
                        
                    if formatted_date not in existing_dates:
                        # Use the first value as the placeholder for missing dates
                        placeholder_value = chart_data[0]["value"] if chart_data else 0
                        chart_data.append({
                            "date": formatted_date,
                            "value": placeholder_value
                        })
                        
                # Resort after adding placeholders
                chart_data = sorted(chart_data, key=lambda x: x["date"])
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error formatting chart data: {str(e)}")
            logger.error(traceback.format_exc())
            return self._get_mock_chart_data(timeframe)
    
    def _get_mock_chart_data(self, timeframe):
        """Generate mock chart data for testing when no real data is available"""
        from datetime import datetime, timedelta
        
        # Generate dates based on timeframe
        end_date = datetime.now()
        days_to_subtract = 1
        
        if timeframe == '1W':
            days_to_subtract = 7
        elif timeframe == '1M':
            days_to_subtract = 30
            
        # Generate 7 data points for the chart
        chart_data = []
        for i in range(7):
            date = end_date - timedelta(days=days_to_subtract * (6-i))
            value = 10000 + (i * 500) + ((i % 3) * 200)  # Generate slightly variable data
            
            # Add some volatility for visual interest
            volatility = (i % 2) * 300
            if i % 3 == 0:
                volatility = -volatility
                
            chart_data.append({
                "timestamp": date.strftime("%Y-%m-%d"),
                "value": value + volatility,
                "pnl": (value + volatility) - 10000,  # PnL based on starting value
                "winRate": 50 + (i * 3),  # Increasing win rate
                "drawdown": max(0, 10 - i)  # Decreasing drawdown
            })
            
        return chart_data
    
    def get_recent_activity(self) -> Dict[str, Any]:
        """Get recent trading activity for the dashboard"""
        try:
            # Get today's date for filtering
            today = datetime.now().date()
            today_start = datetime.combine(today, datetime.min.time())
            
            # Get trades executed today (both open and closed)
            today_trades = db.get_trades_by_date_range(
                start_date=today_start.isoformat(),
                end_date=datetime.now().isoformat()
            )
            
            # If there are no trades from today, check active trades that might have been opened earlier
            if not today_trades:
                active_trades = db.get_active_trades()
                today_trades = active_trades
            
            # Define the last 7 days for weekday organization
            days_of_week = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            current_weekday = datetime.now().weekday()  # 0 = Monday, 6 = Sunday
            
            # Initialize dictionaries for trade counts and volumes
            trades_by_day = {day: 0 for day in days_of_week}
            volume_by_day = {day: 0 for day in days_of_week}
            
            # Get trades for the past 7 days for activity chart
            one_week_ago = (datetime.now() - timedelta(days=7)).isoformat()
            trades_past_week = db.get_trades_by_date_range(
                start_date=one_week_ago,
                end_date=datetime.now().isoformat()
            )
            
            # Go through trades and organize by day of week
            for trade in trades_past_week:
                try:
                    # Parse the open_time
                    open_time = None
                    if isinstance(trade.get("open_time"), str):
                        try:
                            open_time = datetime.fromisoformat(trade.get("open_time").replace('Z', '+00:00'))
                        except ValueError:
                            try:
                                open_time = datetime.strptime(trade.get("open_time"), '%Y-%m-%d %H:%M:%S')
                            except ValueError:
                                continue
                    else:
                        open_time = trade.get("open_time")
                    
                    if not open_time:
                        continue
                    
                    # Get day of week
                    day_name = days_of_week[open_time.weekday()]
                    
                    # Count trade
                    trades_by_day[day_name] += 1
                    
                    # Add volume (scale for visualization)
                    volume = float(trade.get("volume", 0))
                    volume_by_day[day_name] += volume * 100
                except Exception as e:
                    logger.error(f"Error processing trade date: {str(e)}")
            
            # Format data for the chart
            # Make sure the days are in correct order (starting from today and going back 7 days)
            ordered_days = []
            for i in range(7):
                day_idx = (current_weekday - i) % 7
                ordered_days.append(days_of_week[day_idx])
                
            ordered_days.reverse()  # Reverse to get chronological order (past to present)
            
            # Create the activity data with the ordered days
            activity_data = []
            for day in ordered_days:
                activity_data.append({
                    "day": day,
                    "trades": trades_by_day[day],
                    "volume": volume_by_day[day]
                })
            
            # Format the response
            return {
                "todayTrades": len(today_trades),
                "weeklyData": activity_data
            }
        except Exception as e:
            logger.error(f"Error getting recent activity: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return mock data as fallback with better defaults
            return {
                "todayTrades": 0,
                "weeklyData": [
                    {"day": "Mon", "trades": 0, "volume": 0},
                    {"day": "Tue", "trades": 0, "volume": 0},
                    {"day": "Wed", "trades": 0, "volume": 0},
                    {"day": "Thu", "trades": 0, "volume": 0},
                    {"day": "Fri", "trades": 0, "volume": 0},
                    {"day": "Sat", "trades": 0, "volume": 0},
                    {"day": "Sun", "trades": 0, "volume": 0}
                ]
            }
    
    def start_frontend(self) -> bool:
        """Start the Next.js frontend for the dashboard"""
        try:
            if self.frontend_process is not None and self.frontend_process.poll() is None:
                # Frontend is already running
                return True
                
            logger.info("Starting dashboard frontend...")
            
            # Get the base directory
            base_dir = Path(__file__).resolve().parent.parent
            dashboard_dir = base_dir / "trading-dash"
            
            if not dashboard_dir.exists():
                logger.error(f"Dashboard directory not found at {dashboard_dir}")
                return False
                
            # Create .env.local with backend URL
            env_file = dashboard_dir / ".env.local"
            with open(env_file, "w") as f:
                f.write("NEXT_PUBLIC_API_URL=http://localhost:8000\n")
                f.write("NEXT_PUBLIC_WS_URL=ws://localhost:8000\n")
            
            # Start the Next.js dev server
            self.frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=dashboard_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            # Give it a moment to start
            import time
            time.sleep(3)
            
            # Check if it's running
            if self.frontend_process.poll() is not None:
                stderr = self.frontend_process.stderr.read().decode('utf-8')
                logger.error(f"Failed to start frontend: {stderr}")
                return False
                
            logger.info("Dashboard frontend started at http://localhost:3000")
            return True
            
        except Exception as e:
            logger.error(f"Error starting frontend: {str(e)}")
            return False
    
    def get_account_history_data(self) -> Dict[str, Any]:
        """
        Get account history data for the dashboard including trades and balance history
        
        Returns:
            Dict[str, Any]: Account history data including trades and balance history
        """
        try:
            # Initialize result structure
            result = {
                "trades": [],
                "balance_history": []
            }
            
            if not self.trading_bot or not hasattr(self.trading_bot, 'mt5_handler'):
                logger.error("Cannot get account history: Trading bot or MT5 handler not available")
                return self._generate_mock_history_data()
                
            # Get trade history from MT5
            days = 30  # Look back 30 days
            
            # Get deals from MT5
            mt5_handler = self.trading_bot.mt5_handler
            from_date = datetime.now() - timedelta(days=days)
            
            # Get balance history
            logger.info("Fetching balance history from MT5")
            balance_history = mt5_handler.get_account_history(days)
            logger.debug(f"Retrieved {len(balance_history)} balance history records")
            result["balance_history"] = balance_history
            
            # Get closed trades from database
            logger.info("Fetching closed trades from database")
            closed_trades = db.get_closed_trades(limit=100)  # Get last 100 closed trades
            logger.debug(f"Retrieved {len(closed_trades)} closed trades")
            
            # Format trades
            for trade in closed_trades:
                # Try to get profit as a float
                profit = 0
                try:
                    profit = float(trade.get("profit", 0))
                except (ValueError, TypeError):
                    profit = 0
                    
                result["trades"].append({
                    "id": str(trade.get("id", "")),
                    "symbol": trade.get("symbol", ""),
                    "type": trade.get("type", "").lower(),
                    "entry_price": float(trade.get("entry_price", 0)),
                    "exit_price": float(trade.get("exit_price", 0)),
                    "volume": float(trade.get("volume", 0)),
                    "profit": profit,
                    "open_time": trade.get("open_time", ""),
                    "close_time": trade.get("close_time", ""),
                    "stop_loss": float(trade.get("stop_loss", 0)),
                    "take_profit": float(trade.get("take_profit", 0)),
                    "strategy": trade.get("strategy", ""),
                    "duration": trade.get("duration", ""),
                })
            
            # If both lists are empty, generate mock data
            if not result["trades"] and not result["balance_history"]:
                logger.info("No real history data found, generating mock data")
                return self._generate_mock_history_data()
                
            return result
            
        except Exception as e:
            logger.error(f"Error getting account history: {str(e)}")
            logger.error(traceback.format_exc())
            return self._generate_mock_history_data()
            
    def _generate_mock_history_data(self) -> Dict[str, Any]:
        """Generate mock history data for demonstration purposes"""
        try:
            # Check if we have real closed trades first - if so, don't generate mock data
            closed_trades = db.get_closed_trades(limit=10)
            if closed_trades and len(closed_trades) > 0:
                logger.info("Using real closed trades data instead of generating mock data")
                
            # Get real balance history if possible
            balance_history = []
            if self.trading_bot and hasattr(self.trading_bot, 'mt5_handler'):
                try:
                    mt5_handler = self.trading_bot.mt5_handler
                    balance_history = mt5_handler.get_account_history(days=30) or []
                except Exception as e:
                        logger.error(f"Error getting account history: {str(e)}")
                
                # Format the real trades for display
                formatted_trades = []
                for trade in closed_trades:
                    # Format the trade data
                    formatted_trades.append({
                        "id": str(trade.get("id", "")),
                        "ticket": trade.get("ticket", 0),
                        "symbol": trade.get("symbol", ""),
                        "type": trade.get("type", "").lower(),
                        "entry_price": float(trade.get("entry_price", 0)),
                        "exit_price": float(trade.get("exit_price", 0)),
                        "volume": float(trade.get("volume", 0)),
                        "profit": float(trade.get("profit", 0)),
                        "open_time": trade.get("open_time", ""),
                        "close_time": trade.get("close_time", ""),
                        "stop_loss": float(trade.get("stop_loss", 0)),
                        "take_profit": float(trade.get("take_profit", 0)),
                        "strategy": trade.get("strategy", ""),
                        "duration": trade.get("duration", "")
                    })
                
                return {
                    "trades": formatted_trades,
                    "balance_history": balance_history
                }
        
            # Continue with mock data generation if no real data found
            logger.info("Generating mock history data")
            
            # Get MT5 handler if available
            mt5_handler = None
            if self.trading_bot and hasattr(self.trading_bot, 'mt5_handler'):
                mt5_handler = self.trading_bot.mt5_handler
            
            # Get real balance history if possible
            days = 30
            balance_history = []
            
            if mt5_handler:
                real_balance_history = mt5_handler.get_account_history(days=30)
                
                if real_balance_history and len(real_balance_history) > 0:
                    logger.info(f"Using real balance history ({len(real_balance_history)} records)")
                    balance_history = real_balance_history
                    
                    # Generate mock trades that align with the balance history
                    trades = self._generate_trades_from_balance_history(balance_history)
                    
                    return {
                        "trades": trades,
                        "balance_history": balance_history
                    }
            
            # If no real data available, generate completely mock data
            logger.info("Generating mock history data")
            
            # Current date for reference
            end_date = datetime.now()
            
            # Generate 30 days of balance history
            balance_history = []
            current_balance = 1000.0
            total_profit = 0
            
            for i in range(30, 0, -1):
                date = end_date - timedelta(days=i)
                
                # Some random daily change
                if i % 5 == 0:  # More significant changes every 5 days
                    daily_change = (0.5 - random.random()) * 100
                else:
                    daily_change = (0.5 - random.random()) * 20
                    
                current_balance += daily_change
                total_profit += daily_change
                
                # Calculate drawdown from peak
                peak_balance = max([b.get('balance', 0) for b in balance_history] + [current_balance])
                drawdown = 0
                if peak_balance > 0:
                    drawdown = (peak_balance - current_balance) / peak_balance * 100
                
                balance_history.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "balance": round(current_balance, 2),
                    "profit_loss": round(daily_change, 2),
                    "drawdown": round(drawdown, 2),
                    "win_rate": i * 2  # Just for visualization
                })
            
            # Generate some mock trades
            trades = []
            symbols = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD"]
            strategies = ["Turtle Soup", "BMS+RTO", "AMD"]
            
            # Generate more trades to match profit/loss pattern
            num_trades = 15
            winning_trades = int(num_trades * 0.6)  # 60% win rate
            
            for i in range(num_trades):
                # Random date within last 30 days
                days_ago = random.randint(1, 29)
                open_date = end_date - timedelta(days=days_ago)
                
                # Trade duration between 1 hour and 2 days
                duration_hours = random.randint(1, 48)
                close_date = open_date + timedelta(hours=duration_hours)
                
                # Random profit/loss based on win/loss ratio
                is_winner = i < winning_trades
                if is_winner:
                    profit = random.uniform(10, 50)
                else:
                    profit = -random.uniform(10, 40)
                
                # Type based on profit
                trade_type = "buy" if profit > 0 else "sell"
                
                trades.append({
                    "id": str(i + 1),
                    "symbol": random.choice(symbols),
                    "type": trade_type,
                    "entry_price": round(1.0 + random.random() * 0.5, 5),
                    "exit_price": round(1.0 + random.random() * 0.5, 5),
                    "volume": round(0.1 + random.random() * 0.9, 2),
                    "profit": round(profit, 2),
                    "open_time": open_date.isoformat(),
                    "close_time": close_date.isoformat(),
                    "stop_loss": round(1.0 + random.random() * 0.4, 5),
                    "take_profit": round(1.0 + random.random() * 0.6, 5),
                    "strategy": random.choice(strategies),
                    "duration": f"{duration_hours}h",
                })
            
            return {
                "trades": trades,
                "balance_history": balance_history
            }
        except Exception as e:
            logger.error(f"Error generating mock history data: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return minimal mock data
            return {
                "trades": [],
                "balance_history": []
            }
            
    def _generate_trades_from_balance_history(self, balance_history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate realistic trades based on balance history changes"""
        trades = []
        symbols = ["EURUSD", "USDJPY", "GBPUSD", "AUDUSD", "USDCAD"]
        strategies = ["Turtle Soup", "BMS+RTO", "AMD"]
        
        # Group days with similar profit/loss
        trade_days = []
        current_day = None
        
        for day in balance_history:
            profit_loss = day.get("profit_loss", 0)
            
            # Skip days with no change
            if abs(profit_loss) < 0.01:
                continue
                
            if current_day is None:
                current_day = {
                    "date": day["date"],
                    "profit_loss": profit_loss
                }
            elif (profit_loss > 0 and current_day["profit_loss"] > 0) or (profit_loss < 0 and current_day["profit_loss"] < 0):
                # Same direction, aggregate
                current_day["profit_loss"] += profit_loss
            else:
                # Direction changed, save current and start new
                trade_days.append(current_day)
                current_day = {
                    "date": day["date"],
                    "profit_loss": profit_loss
                }
        
        # Add the last day if exists
        if current_day is not None:
            trade_days.append(current_day)
            
        # Generate trades for each trading day
        for i, day in enumerate(trade_days):
            profit_loss = day["profit_loss"]
            
            # Skip tiny changes
            if abs(profit_loss) < 1.0:
                continue
                
            date = datetime.strptime(day["date"], "%Y-%m-%d")
            
            # Random time during trading hours
            hour = random.randint(8, 17)
            minute = random.randint(0, 59)
            open_date = date.replace(hour=hour, minute=minute)
            
            # Trade duration
            duration_hours = random.randint(1, 12)
            close_date = open_date + timedelta(hours=duration_hours)
            
            # Trade type
            trade_type = "buy" if profit_loss > 0 else "sell"
            
            # Entry and exit prices that make sense for profit
            base_price = 1.0 + random.random() * 0.5
            pip_value = 0.0001
            pips = int(abs(profit_loss) / 10)  # Rough estimate
            
            if trade_type == "buy":
                entry_price = base_price
                exit_price = base_price + (pips * pip_value)
            else:
                entry_price = base_price
                exit_price = base_price - (pips * pip_value)
            
            trades.append({
                "id": str(i + 1),
                "symbol": random.choice(symbols),
                "type": trade_type,
                "entry_price": round(entry_price, 5),
                "exit_price": round(exit_price, 5),
                "volume": round(0.1 + random.random() * 0.9, 2),
                "profit": round(profit_loss, 2),
                "open_time": open_date.isoformat(),
                "close_time": close_date.isoformat(),
                "stop_loss": round(entry_price * (0.98 if trade_type == "buy" else 1.02), 5),
                "take_profit": round(entry_price * (1.02 if trade_type == "buy" else 0.98), 5),
                "strategy": random.choice(strategies),
                "duration": f"{duration_hours}h",
            })
            
        return trades
    
    def get_detailed_trade_history(self, limit: int = 100, offset: int = 0, include_active: bool = True) -> Dict[str, Any]:
        """
        Get detailed trade history with profit/loss information
        
        Args:
            limit: Maximum number of trades to return
            offset: Number of trades to skip (for pagination)
            include_active: Whether to include active trades in the results
            
        Returns:
            Dict[str, Any]: Detailed trade history with profit/loss information
        """
        try:
            # Initialize result structure
            result = {
                "trades": [],
                "total_count": 0,
                "summary": {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_profit": 0,
                    "total_loss": 0,
                    "net_pnl": 0,
                    "win_rate": 0,
                    "average_profit": 0,
                    "average_loss": 0,
                    "profit_factor": 0,
                    "largest_win": 0,
                    "largest_loss": 0,
                    "average_duration": "0h 0m"
                }
            }
            
            # Try to get data from MT5 first if available
            all_trades = []
            
            # Check if MT5 handler is available
            mt5_handler = None
            if self.trading_bot and hasattr(self.trading_bot, 'mt5_handler'):
                mt5_handler = self.trading_bot.mt5_handler
            
            if mt5_handler:
                # Get history from MT5 directly - look back 90 days for comprehensive history
                try:
                    logger.info("Fetching trade history directly from MT5")
                    mt5_history = mt5_handler.get_order_history(days=90)
                    
                    # Process closed trades from MT5
                    for trade in mt5_history:
                        # Process trade if it represents a real trade (not a pending order)
                        if trade.get("state") == 3:  # State 3 indicates a filled and closed order
                            # Determine direction based on type
                            trade_type = trade.get("type")
                            direction = "buy" if trade_type in [0, 2, 4] else "sell"  # MT5 trade types
                            
                            # Calculate duration
                            duration_str = "N/A"
                            try:
                                open_time = datetime.fromtimestamp(trade.get("time"))
                                close_time = datetime.fromtimestamp(trade.get("time_close"))
                                duration = close_time - open_time
                                hours, remainder = divmod(duration.total_seconds(), 3600)
                                minutes, _ = divmod(remainder, 60)
                                duration_str = f"{int(hours)}h {int(minutes)}m"
                            except Exception:
                                pass
                            
                            # Determine status
                            profit = float(trade.get("profit", 0))
                            status = "profit" if profit > 0 else "loss"
                            if profit == 0:
                                status = "breakeven"
                            
                            # Calculate profit percentage based on entry and exit prices
                            entry_price = float(trade.get("price", 0))
                            volume = float(trade.get("volume", 0))
                            profit_pct = 0
                            
                            if entry_price > 0 and volume > 0:
                                # Approximate percentage calculation
                                profit_pct = (profit / (entry_price * volume * 100)) * 100
                            
                            formatted_trade = {
                                "id": str(trade.get("ticket", "")),
                                "ticket": trade.get("ticket", ""),
                                "symbol": trade.get("symbol", ""),
                                "direction": direction,
                                "type": direction,
                                "entryPrice": float(trade.get("price", 0)),
                                "exitPrice": float(trade.get("price_current", 0)),
                                "currentPrice": float(trade.get("price_current", 0)),
                                "stopLoss": float(trade.get("sl", 0)),
                                "takeProfit": float(trade.get("tp", 0)),
                                "duration": duration_str,
                                "status": status,
                                "profitLoss": profit,
                                "profitLossPercentage": profit_pct,
                                "size": float(trade.get("volume", 0.0)),
                                "strategy": "MT5 Trade",  # Not available from MT5, use fixed value
                                "openTime": datetime.fromtimestamp(trade.get("time")).isoformat(),
                                "closeTime": datetime.fromtimestamp(trade.get("time_close")).isoformat(),
                                "isActive": False
                            }
                            
                            all_trades.append(formatted_trade)
                    
                    # If include_active is true, also get active positions from MT5
                    if include_active:
                        try:
                            mt5_positions = mt5_handler.get_open_positions()
                            
                            for position in mt5_positions:
                                # Determine direction based on type
                                position_type = position.get("type")
                                direction = "buy" if position_type in [0, 2, 4] else "sell"
                                
                                # Calculate duration so far
                                duration_str = "N/A"
                                try:
                                    open_time = datetime.fromtimestamp(position.get("time"))
                                    now = datetime.now()
                                    duration = now - open_time
                                    hours, remainder = divmod(duration.total_seconds(), 3600)
                                    minutes, _ = divmod(remainder, 60)
                                    duration_str = f"{int(hours)}h {int(minutes)}m"
                                except Exception:
                                    pass
                                
                                # Determine status based on unrealized profit
                                profit = float(position.get("profit", 0))
                                status = "open"
                                
                                # Approximate profit percentage
                                entry_price = float(position.get("price_open", 0))
                                current_price = float(position.get("price_current", 0))
                                volume = float(position.get("volume", 0))
                                profit_pct = 0
                                
                                if entry_price > 0 and current_price > 0 and volume > 0:
                                    if direction == "buy":
                                        profit_pct = ((current_price - entry_price) / entry_price) * 100
                                    else:
                                        profit_pct = ((entry_price - current_price) / entry_price) * 100
                                
                                formatted_trade = {
                                    "id": str(position.get("ticket", "")),
                                    "ticket": position.get("ticket", ""),
                                    "symbol": position.get("symbol", ""),
                                    "direction": direction,
                                    "type": direction,
                                    "entryPrice": entry_price,
                                    "exitPrice": 0,  # Open position
                                    "currentPrice": current_price,
                                    "stopLoss": float(position.get("sl", 0)),
                                    "takeProfit": float(position.get("tp", 0)),
                                    "duration": duration_str,
                                    "status": status,
                                    "profitLoss": profit,
                                    "profitLossPercentage": profit_pct,
                                    "size": volume,
                                    "strategy": "MT5 Active Trade",  # Not available from MT5
                                    "openTime": datetime.fromtimestamp(position.get("time")).isoformat(),
                                    "closeTime": "Active",
                                    "isActive": True
                                }
                                
                                all_trades.append(formatted_trade)
                        except Exception as e:
                            logger.error(f"Error getting active positions from MT5: {str(e)}")
                    
                    # If we have MT5 data, no need to check the database
                    if all_trades:
                        logger.info(f"Got {len(all_trades)} trades from MT5 directly")
                    else:
                        logger.warning("No trades found from MT5, will fallback to database")
                except Exception as e:
                    logger.error(f"Error getting trades from MT5: {str(e)}")
                    logger.error(traceback.format_exc())
            
            # Fallback to database if MT5 didn't provide any trades
            if not all_trades:
                logger.info("Falling back to database for trade history")
                
                # Get closed trades from database
                closed_trades = db.get_closed_trades(limit=1000)  # Get a large number to calculate stats properly
                logger.debug(f"Retrieved {len(closed_trades)} closed trades from database")
                
                # Format closed trades (keep existing database code)
                for trade in closed_trades:
                    # Existing code for processing database trades
                    # ... (existing formatting code)
                    
                    # Try to get profit as a float
                    profit = 0
                    try:
                        profit = float(trade.get("profit_loss", 0))
                    except (ValueError, TypeError):
                        profit = 0
                    
                    # Calculate profit percentage
                    entry_price = float(trade.get("entry_price", 0))
                    exit_price = float(trade.get("current_price", 0))
                    profit_pct = 0
                    
                    if entry_price > 0:
                        direction = trade.get("direction", "").lower()
                        if direction == "buy":
                            profit_pct = ((exit_price - entry_price) / entry_price) * 100
                        elif direction == "sell":
                            profit_pct = ((entry_price - exit_price) / entry_price) * 100
                    
                    # Calculate duration
                    duration_str = "N/A"
                    try:
                        open_time = datetime.fromisoformat(trade.get("open_time", "").replace('Z', '+00:00'))
                        close_time = datetime.fromisoformat(trade.get("close_time", "").replace('Z', '+00:00'))
                        duration = close_time - open_time
                        hours, remainder = divmod(duration.total_seconds(), 3600)
                        minutes, _ = divmod(remainder, 60)
                        duration_str = f"{int(hours)}h {int(minutes)}m"
                    except Exception:
                        pass
                    
                    # Determine status
                    status = "profit" if profit > 0 else "loss"
                    if profit == 0:
                        status = "breakeven"
                    
                    formatted_trade = {
                        "id": str(trade.get("id", "")),
                        "ticket": trade.get("ticket", ""),
                        "symbol": trade.get("symbol", ""),
                        "direction": trade.get("direction", "").lower(),
                        "type": trade.get("direction", "").lower(),
                        "entryPrice": float(trade.get("entry_price", 0)),
                        "exitPrice": float(trade.get("current_price", 0)),
                        "currentPrice": float(trade.get("current_price", 0)),
                        "stopLoss": float(trade.get("stop_loss", 0)),
                        "takeProfit": float(trade.get("take_profit", 0)),
                        "duration": duration_str,
                        "status": status,
                        "profitLoss": profit,
                        "profitLossPercentage": profit_pct,
                        "size": float(trade.get("position_size", 0.0)),
                        "strategy": "Database Trade",  # Not available in database
                        "openTime": trade.get("open_time", ""),
                        "closeTime": trade.get("close_time", "Active"),
                        "isActive": False
                    }
                    
                    all_trades.append(formatted_trade)
                
                # Add active trades if requested
                if include_active:
                    active_trades = db.get_active_trades()
                    logger.debug(f"Retrieved {len(active_trades)} active trades from database")
                    
                    # Format active trades (existing code)
                    for trade in active_trades:
                        # ... (existing active trade formatting code)
                        formatted_trade = {
                            "id": str(trade.get("id", "")),
                            "ticket": trade.get("ticket", ""),
                            "symbol": trade.get("symbol", ""),
                            "direction": trade.get("direction", "").lower(),
                            "type": trade.get("direction", "").lower(),
                            "entryPrice": float(trade.get("entry_price", 0)),
                            "exitPrice": 0,
                            "currentPrice": float(trade.get("current_price", 0)),
                            "stopLoss": float(trade.get("stop_loss", 0)),
                            "takeProfit": float(trade.get("take_profit", 0)),
                            "duration": "Active",
                            "status": "open",
                            "profitLoss": 0, # Not available for open trades
                            "profitLossPercentage": 0,
                            "size": float(trade.get("position_size", 0.0)),
                            "strategy": "DB Active Trade",
                            "openTime": trade.get("open_time", ""),
                            "closeTime": "Active",
                            "isActive": True
                        }
                        
                        all_trades.append(formatted_trade)
            
            # Sort trades by open time (newest first)
            all_trades.sort(key=lambda x: x.get("openTime", ""), reverse=True)
            
            # Calculate summary statistics
            total_trades = len(all_trades)
            winning_trades = sum(1 for t in all_trades if t["status"] == "profit")
            losing_trades = sum(1 for t in all_trades if t["status"] == "loss")
            
            # Calculate profit statistics using profitLoss field
            total_profit = sum(t["profitLoss"] for t in all_trades if t["profitLoss"] > 0)
            total_loss = abs(sum(t["profitLoss"] for t in all_trades if t["profitLoss"] < 0))
            net_pnl = total_profit - total_loss
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            average_profit = (total_profit / winning_trades) if winning_trades > 0 else 0
            average_loss = (total_loss / losing_trades) if losing_trades > 0 else 0
            
            profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
            
            # Find largest win and loss using profitLoss field
            largest_win = max([t["profitLoss"] for t in all_trades if t["profitLoss"] > 0], default=0)
            largest_loss = abs(min([t["profitLoss"] for t in all_trades if t["profitLoss"] < 0], default=0))
            
            # Calculate average duration (for closed trades only)
            closed_trades_list = [t for t in all_trades if not t.get("isActive", False)]
            if closed_trades_list:
                # Calculate average duration in minutes
                total_duration_minutes = 0
                count_with_duration = 0
                
                for trade in closed_trades_list:
                    try:
                        duration_str = trade["duration"]
                        if duration_str != "N/A":
                            # Parse the duration string (format: "Xh Ym")
                            hours = int(duration_str.split('h')[0])
                            minutes = int(duration_str.split('h')[1].strip().split('m')[0])
                            total_duration_minutes += (hours * 60 + minutes)
                            count_with_duration += 1
                    except Exception:
                        pass
                
                if count_with_duration > 0:
                    avg_minutes = total_duration_minutes / count_with_duration
                    avg_hours, avg_minutes_remainder = divmod(avg_minutes, 60)
                    result["summary"]["average_duration"] = f"{int(avg_hours)}h {int(avg_minutes_remainder)}m"
            
            # Update summary data
            result["summary"].update({
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "total_profit": total_profit,
                "total_loss": total_loss,
                "net_pnl": net_pnl,
                "win_rate": win_rate,
                "average_profit": average_profit,
                "average_loss": average_loss,
                "profit_factor": profit_factor,
                "largest_win": largest_win,
                "largest_loss": largest_loss
            })
            
            # Apply pagination
            result["total_count"] = total_trades
            result["trades"] = all_trades[offset:offset+limit]
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting detailed trade history: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Return empty result on error
            return {
                "trades": [],
                "total_count": 0,
                "summary": {
                    "total_trades": 0,
                    "winning_trades": 0,
                    "losing_trades": 0,
                    "total_profit": 0,
                    "total_loss": 0,
                    "net_pnl": 0,
                    "win_rate": 0,
                    "average_profit": 0,
                    "average_loss": 0,
                    "profit_factor": 0,
                    "largest_win": 0,
                    "largest_loss": 0,
                    "average_duration": "0h 0m"
                }
            }
    
    def run(self, host="0.0.0.0", port=8000):
        """Run the FastAPI server"""
        try:
            logger.info(f"Starting dashboard API server on {host}:{port}")
            
            # Get the path to the current Python executable
            python_executable = sys.executable
            
            # Get project root path as a string with proper escaping
            project_root = str(Path(__file__).resolve().parent.parent)
            
            # Create a marker file that will indicate successful startup
            success_marker_file = Path(project_root) / "dashboard_api_running.marker"
            
            script_content = f"""
import sys
from pathlib import Path
import os

# Set environment variable to indicate this is a subprocess
os.environ["DASHBOARD_API_SUBPROCESS"] = "1"

# Add project root to path using pathlib to avoid escape issues
sys.path.insert(0, r'''{project_root}''')

# Import FastAPI components
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from typing import List, Dict, Any, Optional

# Create a simple FastAPI app with the same routes
app = FastAPI(title="Trading Bot Dashboard API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    # Initialize MT5Handler for data access
    from src.mt5_handler import MT5Handler
    mt5_handler = MT5Handler()
    if mt5_handler.initialize():
        # Create success marker file to indicate successful startup
        success_marker_path = r'''{str(success_marker_file)}'''
        Path(success_marker_path).touch()
        print(f"Created success marker at {{success_marker_path}}")
    else:
        print("Failed to initialize MT5Handler")
except Exception as e:
    print(f"Error initializing MT5: {{e}}")
    sys.exit(1)

# Routes
@app.get("/api/dashboard/overview")
async def get_overview():
    account_info = mt5_handler.get_account_info()
    return {{
        "balance": account_info.get("balance", 0),
        "equity": account_info.get("equity", 0),
        "freeMargin": account_info.get("free_margin", 0),
        "maxDrawdown": 0.0,
        "riskLevel": "Low",
        "dailyPnL": account_info.get("equity", 0) - account_info.get("balance", 0),
        "openPositions": len(mt5_handler.get_open_positions())
    }}

@app.get("/api/dashboard/active-trades")
async def get_active_trades():
    positions = mt5_handler.get_open_positions()
    result = []
    for position in positions:
        result.append({{
            "id": str(position.get("ticket", "")),
            "symbol": position.get("symbol", ""),
            "type": "buy" if position.get("type") == 0 else "sell",
            "entryPrice": position.get("price_open", 0),
            "currentPrice": position.get("price_current", 0),
            "stopLoss": position.get("sl", 0),
            "takeProfit": position.get("tp", 0),
            "duration": "0h 0m",
            "status": "profit" if position.get("profit", 0) > 0 else "loss",
            "profitLoss": position.get("profit", 0),
            "profitLossPercentage": 0.0
        }})
    return result

@app.get("/api/dashboard/performance")
async def get_performance():
    # Get the requested timeframe from the query parameter, default to 1D
    timeframe = app.request.query_params.get("timeframe", "1D")
    return {{
        "totalTrades": 0,
        "winRate": 0,
        "profitLoss": 0,
        "profitLossPercentage": 0,
        "portfolioValue": 0,
        "portfolioGrowth": 0,
        "timeframe": timeframe,
        "chartData": []
    }}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Wait a small amount to not consume CPU
            await asyncio.sleep(1)
            
            # Send basic data
            account_info = mt5_handler.get_account_info()
            await websocket.send_json({{
                "type": "account_overview",
                "data": {{
                    "balance": account_info.get("balance", 0),
                    "equity": account_info.get("equity", 0),
                    "freeMargin": account_info.get("free_margin", 0),
                    "maxDrawdown": 0.0,
                    "riskLevel": "Low",
                    "dailyPnL": account_info.get("equity", 0) - account_info.get("balance", 0),
                    "openPositions": len(mt5_handler.get_open_positions())
                }}
            }})
    except WebSocketDisconnect:
        pass
    except Exception as e:
        import logging
        logging.error(f"WebSocket error: {{e}}")

# Run
uvicorn.run(app, host="{host}", port={port})
"""
            # Write the script to a temporary file
            script_path = Path(__file__).resolve().parent.parent / "temp_dashboard_server.py"
            with open(script_path, "w") as f:
                f.write(script_content)
            
            # Start the server as a separate process
            env = os.environ.copy()
            env["DASHBOARD_API_SUBPROCESS"] = "1"
            
            self._api_process = subprocess.Popen(
                [python_executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env
            )
            
            # Wait briefly to see if the process starts successfully
            import time
            time.sleep(1)
            
            if self._api_process.poll() is not None:
                # Process has already terminated - there was an error
                stderr = self._api_process.stderr.read().decode('utf-8')
                logger.error(f"Failed to start API server: {stderr}")
                raise RuntimeError(f"Failed to start API server: {stderr}")
            
            logger.info(f"Dashboard API server running on {host}:{port}")
            
            # Remove any existing marker file
            if success_marker_file.exists():
                success_marker_file.unlink()
            
        except Exception as e:
            logger.error(f"Error running dashboard API: {str(e)}")
            raise
        
    def shutdown(self):
        """Shut down the dashboard API including the frontend process"""
        try:
            # Terminate API server process if running
            if hasattr(self, '_api_process') and self._api_process and self._api_process.poll() is None:
                logger.info("Terminating API server process...")
                self._api_process.terminate()
                self._api_process = None
            
            # Remove temporary script file
            script_path = Path(__file__).resolve().parent.parent / "temp_dashboard_server.py"
            if script_path.exists():
                try:
                    script_path.unlink()
                except:
                    pass
                    
            # Remove marker file
            marker_path = Path(__file__).resolve().parent.parent / "dashboard_api_running.marker"
            if marker_path.exists():
                try:
                    marker_path.unlink()
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Error shutting down API server: {str(e)}")
        
        try:
            # Terminate frontend process if running
            if self.frontend_process and self.frontend_process.poll() is None:
                logger.info("Terminating frontend process...")
                self.frontend_process.terminate()
                self.frontend_process = None
        except Exception as e:
            logger.error(f"Error terminating frontend process: {str(e)}")
        
        if self._owns_trading_bot:
            # Only shutdown the trading bot if we created it
            asyncio.create_task(self.trading_bot.stop())

# Run the server if executed directly
if __name__ == "__main__":
    import uvicorn
    import os
    
    # If run manually, create a simple dashboard with MT5Handler directly
    logger.info("Starting standalone dashboard API server...")
    
    # Create a new FastAPI app
    from fastapi import FastAPI
    app = FastAPI(title="Trading Bot Dashboard API - Standalone")
    
    # Setup CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize MT5Handler
    from src.mt5_handler import MT5Handler
    mt5_handler = MT5Handler()
    mt5_handler.initialize()
    
    # Create basic routes
    @app.get("/api/dashboard/overview")
    async def get_overview():
        account_info = mt5_handler.get_account_info()
        return {
            "balance": account_info.get("balance", 0),
            "equity": account_info.get("equity", 0),
            "freeMargin": account_info.get("free_margin", 0),
            "maxDrawdown": 0.0,
            "riskLevel": "Low",
            "dailyPnL": account_info.get("equity", 0) - account_info.get("balance", 0),
            "openPositions": len(mt5_handler.get_open_positions())
        }
    
    @app.get("/api/dashboard/active-trades")
    async def get_active_trades():
        positions = mt5_handler.get_open_positions()
        result = []
        for position in positions:
            result.append({
                "id": str(position.get("ticket", "")),
                "symbol": position.get("symbol", ""),
                "type": "buy" if position.get("type") == 0 else "sell",
                "entryPrice": position.get("price_open", 0),
                "currentPrice": position.get("price_current", 0),
                "stopLoss": position.get("sl", 0),
                "takeProfit": position.get("tp", 0),
                "duration": "0h 0m",
                "status": "profit" if position.get("profit", 0) > 0 else "loss",
                "profitLoss": position.get("profit", 0),
                "profitLossPercentage": 0.0
            })
        return result
    
    @app.get("/api/dashboard/performance")
    async def get_performance(timeframe: str = "1D"):
        # Get the timeframe from the query parameter with a default of 1D
        return {
            "totalTrades": 0,
            "winRate": 0,
            "profitLoss": 0,
            "profitLossPercentage": 0,
            "portfolioValue": 0,
            "portfolioGrowth": 0,
            "timeframe": timeframe,
            "chartData": {
                "daily": [],
                "weekly": [],
                "monthly": []
            }
        }
    
    # Start a simple frontend if needed
    from pathlib import Path
    dashboard_dir = Path(__file__).resolve().parent.parent / "trading-dash"
    
    if dashboard_dir.exists():
        # Create .env.local with backend URL
        env_file = dashboard_dir / ".env.local"
        with open(env_file, "w") as f:
            f.write("NEXT_PUBLIC_API_URL=http://localhost:8000\n")
            f.write("NEXT_PUBLIC_WS_URL=ws://localhost:8000\n")
        
        # Start frontend in a separate process
        try:
            import subprocess
            frontend_process = subprocess.Popen(
                ["npm", "run", "dev"],
                cwd=dashboard_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info("Dashboard frontend started at http://localhost:3000")
        except Exception as e:
            logger.error(f"Error starting frontend: {e}")
    
    # Run the server
    logger.info("Starting dashboard API server at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 