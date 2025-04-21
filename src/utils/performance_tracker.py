import traceback
from typing import Dict, List, Any
from datetime import datetime
from src.mt5_handler import MT5Handler
from loguru import logger

class PerformanceTracker:
    """
    Handles performance tracking and metrics for the trading bot.
    
    This class is responsible for:
    - Tracking trade performance metrics
    - Calculating win/loss ratios, drawdowns, etc.
    - Generating performance reports
    """
    
    def __init__(self, mt5_handler=None, config=None):
        """
        Initialize the PerformanceTracker.
        
        Args:
            mt5_handler: MT5Handler instance for accessing trade data
            config: Configuration dictionary
        """
        self.mt5_handler = mt5_handler if mt5_handler is not None else MT5Handler()
        self.config = config or {}
        
        # Performance metrics tracking
        self.metrics = {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "total_profit": 0.0,
            "total_loss": 0.0,
            "total_profit_winning": 0.0,
            "total_profit_losing": 0.0,
            "current_drawdown": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "last_updated": datetime.now(),
            "tp_hits": 0,
            "sl_hits": 0,
            "manual_closures": 0,
            "tp_hit_rate": 0.0,
            # Add signal quality metrics
            "avg_signal_quality": 0.0,
            "high_quality_trades": 0,  # Trades with quality > 80%
            "medium_quality_trades": 0,  # Trades with quality 50-80%
            "low_quality_trades": 0,  # Trades with quality < 50%
            "high_quality_win_rate": 0.0,
            "medium_quality_win_rate": 0.0,
            "low_quality_win_rate": 0.0
        }
        
        # Historical performance data
        self.daily_performance = {}
        self.weekly_performance = {}
        self.monthly_performance = {}
        
    def set_mt5_handler(self, mt5_handler):
        """Set the MT5Handler instance after initialization."""
        self.mt5_handler = mt5_handler
    
    async def update_performance_metrics(self) -> Dict[str, Any]:
        """
        Update performance metrics based on trading history.
        
        Returns:
            Dictionary containing updated performance metrics
        """
        try:
            if not self.mt5_handler:
                logger.warning("MT5Handler not set, cannot update performance metrics")
                return self.metrics
                
            # Get trading history from MT5
            logger.info("Fetching trading history for performance metrics...")
            history = self.mt5_handler.get_order_history(days=3)
            
            if not history:
                logger.info("No trading history found for performance metrics")
                
                # Try with a longer period (7 days)
                logger.info("Trying with a 7-day lookback period...")
                extended_history = self.mt5_handler.get_order_history(days=7)
                
                if extended_history:
                    logger.info(f"Found {len(extended_history)} trade records with 7-day lookback")
                    history = extended_history
                else:
                    # Get current open positions for reference
                    open_positions = self.mt5_handler.get_open_positions()
                    logger.info(f"Currently have {len(open_positions)} open positions")
                    
                    # No history available, return current metrics
                    self.metrics["last_updated"] = datetime.now()
                    return self.metrics
                
            # Reset counters
            winning_trades = 0
            losing_trades = 0
            total_profit = 0.0
            total_loss = 0.0
            total_profit_winning = 0.0  # Track winning trades profit
            total_profit_losing = 0.0   # Track losing trades profit (negative value)
            profits = []
            losses = []
            tp_hits = 0
            sl_hits = 0
            manual_closures = 0
            
            logger.info(f"Processing {len(history)} trade history records...")
            
            # Process history entries
            for trade in history:
                profit = trade.get("profit", 0.0)
                signal_quality = trade.get("signal_quality", 0.0)
                
                if profit > 0:
                    winning_trades += 1
                    total_profit += profit
                    total_profit_winning += profit  # Add to winning profit total
                    profits.append(profit)
                elif profit < 0:
                    losing_trades += 1
                    total_loss += abs(profit)  # Store as positive value
                    total_profit_losing += profit  # Keep as negative value
                    losses.append(abs(profit))
                
                # Add closure reason analysis
                reason = trade.get("reason", "unknown")
                if reason == "tp":  # Actual code may differ
                    tp_hits += 1
                elif reason == "sl":
                    sl_hits += 1
                elif reason == "manual":
                    manual_closures += 1
                
                # Track signal quality metrics
                if signal_quality > 0:
                    self.metrics["avg_signal_quality"] += signal_quality
                    if signal_quality >= 0.8:  # 80%+
                        self.metrics["high_quality_trades"] += 1
                        if profit > 0:
                            self.metrics["high_quality_win_rate"] += 1
                    elif signal_quality >= 0.5:  # 50-80%
                        self.metrics["medium_quality_trades"] += 1
                        if profit > 0:
                            self.metrics["medium_quality_win_rate"] += 1
                    else:  # < 50%
                        self.metrics["low_quality_trades"] += 1
                        if profit > 0:
                            self.metrics["low_quality_win_rate"] += 1
            
            # Calculate performance metrics
            total_trades = winning_trades + losing_trades
            
            # Update metrics dictionary
            self.metrics["total_trades"] = total_trades
            self.metrics["winning_trades"] = winning_trades
            self.metrics["losing_trades"] = losing_trades
            self.metrics["total_profit"] = total_profit
            self.metrics["total_loss"] = total_loss
            self.metrics["total_profit_winning"] = total_profit_winning
            self.metrics["total_profit_losing"] = total_profit_losing
            
            # Calculate derived metrics
            if total_trades > 0:
                self.metrics["win_rate"] = winning_trades / total_trades
            
            if total_loss > 0:
                self.metrics["profit_factor"] = total_profit / total_loss
            
            if profits:
                self.metrics["avg_profit"] = sum(profits) / len(profits)
            
            if losses:
                self.metrics["avg_loss"] = sum(losses) / len(losses)
            
            # Calculate expectancy
            if self.metrics["avg_loss"] > 0:
                self.metrics["expectancy"] = (
                    self.metrics["win_rate"] * (self.metrics["avg_profit"] / self.metrics["avg_loss"])
                    - (1 - self.metrics["win_rate"])
                )
            
            # Calculate drawdown (simplified version)
            logger.info("Fetching account history for drawdown calculation...")
            balances = self.mt5_handler.get_account_history(days=3)
            if balances:
                max_equity = 0
                max_drawdown = 0
                current_drawdown = 0
                
                for entry in balances:
                    equity = entry.get("equity", 0)
                    
                    if equity > max_equity:
                        max_equity = equity
                        current_drawdown = 0
                    else:
                        current_drawdown = (max_equity - equity) / max_equity if max_equity > 0 else 0
                        
                    if current_drawdown > max_drawdown:
                        max_drawdown = current_drawdown
                
                self.metrics["current_drawdown"] = current_drawdown
                self.metrics["max_drawdown"] = max_drawdown
            
            # Update timestamp
            self.metrics["last_updated"] = datetime.now()
            
            # Update period performance
            self._update_period_performance(history)
            
            # Update additional metrics
            self.metrics["tp_hits"] = tp_hits
            self.metrics["sl_hits"] = sl_hits
            self.metrics["manual_closures"] = manual_closures
            
            # Calculate TP hit rate if we have any TP or SL hits
            if tp_hits + sl_hits > 0:
                self.metrics["tp_hit_rate"] = tp_hits / (tp_hits + sl_hits)
            
            # Update signal quality metrics
            if self.metrics["avg_signal_quality"] > 0:
                self.metrics["avg_signal_quality"] /= (self.metrics["high_quality_trades"] + self.metrics["medium_quality_trades"] + self.metrics["low_quality_trades"])
            
            logger.info(f"Performance metrics updated: Win rate {self.metrics['win_rate']:.2f}, "
                        f"Profit factor {self.metrics['profit_factor']:.2f}")
            
            return self.metrics
            
        except Exception as e:
            logger.error(f"Error in update_performance_metrics: {str(e)}")
            logger.error(traceback.format_exc())
            return self.metrics
    
    def _update_period_performance(self, history: List[Dict]) -> None:
        """
        Update daily, weekly, and monthly performance data.
        
        Args:
            history: List of trade history entries
        """
        try:
            daily_data = {}
            weekly_data = {}
            monthly_data = {}
            
            # Track quality metrics
            quality_metrics = {
                "high": {"wins": 0, "total": 0},
                "medium": {"wins": 0, "total": 0},
                "low": {"wins": 0, "total": 0}
            }
            total_quality = 0.0
            trades_with_quality = 0
            
            for trade in history:
                # Extract datetime and profit
                close_time = trade.get("close_time")
                if not close_time:
                    continue
                    
                profit = trade.get("profit", 0.0)
                signal_quality = trade.get("signal_quality", 0.0)
                
                # Track signal quality metrics
                if signal_quality > 0:
                    total_quality += signal_quality
                    trades_with_quality += 1
                    
                    # Categorize trade by quality
                    if signal_quality >= 0.8:  # 80%+
                        quality_metrics["high"]["total"] += 1
                        if profit > 0:
                            quality_metrics["high"]["wins"] += 1
                    elif signal_quality >= 0.5:  # 50-80%
                        quality_metrics["medium"]["total"] += 1
                        if profit > 0:
                            quality_metrics["medium"]["wins"] += 1
                    else:  # < 50%
                        quality_metrics["low"]["total"] += 1
                        if profit > 0:
                            quality_metrics["low"]["wins"] += 1
                
                # Create date keys
                date_key = close_time.strftime("%Y-%m-%d")
                week_key = close_time.strftime("%Y-W%W")
                month_key = close_time.strftime("%Y-%m")
                
                # Update daily data
                if date_key not in daily_data:
                    daily_data[date_key] = {"profit": 0.0, "trades": 0, "avg_quality": 0.0, "quality_trades": 0}
                daily_data[date_key]["profit"] += profit
                daily_data[date_key]["trades"] += 1
                if signal_quality > 0:
                    daily_data[date_key]["avg_quality"] = ((daily_data[date_key]["avg_quality"] * daily_data[date_key]["quality_trades"]) + signal_quality) / (daily_data[date_key]["quality_trades"] + 1)
                    daily_data[date_key]["quality_trades"] += 1
                
                # Update weekly data
                if week_key not in weekly_data:
                    weekly_data[week_key] = {"profit": 0.0, "trades": 0, "avg_quality": 0.0, "quality_trades": 0}
                weekly_data[week_key]["profit"] += profit
                weekly_data[week_key]["trades"] += 1
                if signal_quality > 0:
                    weekly_data[week_key]["avg_quality"] = ((weekly_data[week_key]["avg_quality"] * weekly_data[week_key]["quality_trades"]) + signal_quality) / (weekly_data[week_key]["quality_trades"] + 1)
                    weekly_data[week_key]["quality_trades"] += 1
                
                # Update monthly data
                if month_key not in monthly_data:
                    monthly_data[month_key] = {"profit": 0.0, "trades": 0, "avg_quality": 0.0, "quality_trades": 0}
                monthly_data[month_key]["profit"] += profit
                monthly_data[month_key]["trades"] += 1
                if signal_quality > 0:
                    monthly_data[month_key]["avg_quality"] = ((monthly_data[month_key]["avg_quality"] * monthly_data[month_key]["quality_trades"]) + signal_quality) / (monthly_data[month_key]["quality_trades"] + 1)
                    monthly_data[month_key]["quality_trades"] += 1
            
            # Update metrics with quality statistics
            if trades_with_quality > 0:
                self.metrics["avg_signal_quality"] = total_quality / trades_with_quality
                
                self.metrics["high_quality_trades"] = quality_metrics["high"]["total"]
                self.metrics["medium_quality_trades"] = quality_metrics["medium"]["total"]
                self.metrics["low_quality_trades"] = quality_metrics["low"]["total"]
                
                self.metrics["high_quality_win_rate"] = quality_metrics["high"]["wins"] / quality_metrics["high"]["total"] if quality_metrics["high"]["total"] > 0 else 0.0
                self.metrics["medium_quality_win_rate"] = quality_metrics["medium"]["wins"] / quality_metrics["medium"]["total"] if quality_metrics["medium"]["total"] > 0 else 0.0
                self.metrics["low_quality_win_rate"] = quality_metrics["low"]["wins"] / quality_metrics["low"]["total"] if quality_metrics["low"]["total"] > 0 else 0.0
            
            # Store the updated data
            self.daily_performance = daily_data
            self.weekly_performance = weekly_data
            self.monthly_performance = monthly_data
            
        except Exception as e:
            logger.error(f"Error in _update_period_performance: {str(e)}")
    
    async def generate_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.
        
        Returns:
            Dictionary containing performance report data
        """
        # Update metrics first
        await self.update_performance_metrics()
        
        # Create the report
        report = {
            "overall_metrics": self.metrics.copy(),
            "daily_performance": dict(sorted(self.daily_performance.items(), reverse=True)[:7]),  # Last 7 days
            "weekly_performance": dict(sorted(self.weekly_performance.items(), reverse=True)[:4]),  # Last 4 weeks
            "monthly_performance": dict(sorted(self.monthly_performance.items(), reverse=True)[:3]),  # Last 3 months
            "signal_quality_metrics": {
                "average_quality": self.metrics["avg_signal_quality"] * 100,  # Convert to percentage
                "high_quality": {
                    "trades": self.metrics["high_quality_trades"],
                    "win_rate": self.metrics["high_quality_win_rate"] * 100
                },
                "medium_quality": {
                    "trades": self.metrics["medium_quality_trades"],
                    "win_rate": self.metrics["medium_quality_win_rate"] * 100
                },
                "low_quality": {
                    "trades": self.metrics["low_quality_trades"],
                    "win_rate": self.metrics["low_quality_win_rate"] * 100
                }
            },
            "generated_at": datetime.now()
        }
        
        return report 