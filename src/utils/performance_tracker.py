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
        
        # Performance metrics tracking - RESTRUCTURED
        # Now holds metrics per strategy, plus a "Global" aggregate.
        self.metrics = {
            "Global": self._get_new_metrics_dict()
        }
        
        # Historical performance data (can also be made per-strategy if needed later)
        self.daily_performance = {}
        self.weekly_performance = {}
        self.monthly_performance = {}
        
    def _get_new_metrics_dict(self) -> Dict[str, Any]:
        """Returns a fresh dictionary for tracking metrics."""
        return {
            "total_trades": 0, "winning_trades": 0, "losing_trades": 0,
            "total_profit": 0.0, "total_loss": 0.0, "total_profit_winning": 0.0,
            "total_profit_losing": 0.0, "current_drawdown": 0.0, "max_drawdown": 0.0,
            "win_rate": 0.0, "profit_factor": 0.0, "avg_profit": 0.0,
            "avg_loss": 0.0, "expectancy": 0.0, "last_updated": datetime.now(),
            "tp_hits": 0, "sl_hits": 0, "manual_closures": 0, "tp_hit_rate": 0.0,
            "avg_signal_quality": 0.0, "high_quality_trades": 0,
            "medium_quality_trades": 0, "low_quality_trades": 0,
            "high_quality_win_rate": 0.0, "medium_quality_win_rate": 0.0,
            "low_quality_win_rate": 0.0
        }

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
            history = self.mt5_handler.get_order_history(days=30) # Increased lookback
            
            if not history:
                logger.info("No trading history found for performance metrics.")
                # Return current metrics without updating if no history
                return self.metrics
                
            # --- PER-STRATEGY TRACKING ---
            # Reset all strategy metrics before recalculating
            self.metrics = {"Global": self._get_new_metrics_dict()}
            
            logger.info(f"Processing {len(history)} trade history records...")
            
            # Temporary dicts to hold raw numbers for each strategy
            strategy_data = {}

            # Process history entries
            for trade in history:
                profit = trade.get("profit", 0.0)
                strategy_name = trade.get("comment", "Unspecified")

                # Initialize a data dict for the strategy if it's new
                if strategy_name not in strategy_data:
                    strategy_data[strategy_name] = {
                        "profits": [], "losses": [], "winning_trades": 0, "losing_trades": 0
                    }
                
                if profit > 0:
                    strategy_data[strategy_name]['winning_trades'] += 1
                    strategy_data[strategy_name]['profits'].append(profit)
                elif profit < 0:
                    strategy_data[strategy_name]['losing_trades'] += 1
                    strategy_data[strategy_name]['losses'].append(abs(profit))

            # --- AGGREGATE AND CALCULATE METRICS FOR EACH STRATEGY ---
            for strategy_name, data in strategy_data.items():
                if strategy_name not in self.metrics:
                    self.metrics[strategy_name] = self._get_new_metrics_dict()

                stats = self.metrics[strategy_name]
                global_stats = self.metrics["Global"]

                winning_trades = data['winning_trades']
                losing_trades = data['losing_trades']
                total_trades = winning_trades + losing_trades
                
                total_profit = sum(data['profits'])
                total_loss = sum(data['losses'])

                # Update strategy-specific stats
                stats['total_trades'] = total_trades
                stats['winning_trades'] = winning_trades
                stats['losing_trades'] = losing_trades
                stats['total_profit'] = total_profit
                stats['total_loss'] = total_loss
                stats['total_profit_winning'] = total_profit
                stats['total_profit_losing'] = -total_loss
                stats['win_rate'] = winning_trades / total_trades if total_trades > 0 else 0
                stats['profit_factor'] = total_profit / total_loss if total_loss > 0 else 0
                stats['avg_profit'] = total_profit / winning_trades if winning_trades > 0 else 0
                stats['avg_loss'] = total_loss / losing_trades if losing_trades > 0 else 0
                stats['last_updated'] = datetime.now()

                # Update Global stats
                global_stats['total_trades'] += total_trades
                global_stats['winning_trades'] += winning_trades
                global_stats['losing_trades'] += losing_trades
                global_stats['total_profit'] += total_profit
                global_stats['total_loss'] += total_loss
                global_stats['total_profit_winning'] += total_profit
                global_stats['total_profit_losing'] -= total_loss

            # --- FINAL GLOBAL CALCULATIONS ---
            if self.metrics['Global']['total_trades'] > 0:
                self.metrics['Global']['win_rate'] = self.metrics['Global']['winning_trades'] / self.metrics['Global']['total_trades']
            if self.metrics['Global']['total_loss'] > 0:
                self.metrics['Global']['profit_factor'] = self.metrics['Global']['total_profit'] / self.metrics['Global']['total_loss']
            if self.metrics['Global']['winning_trades'] > 0:
                self.metrics['Global']['avg_profit'] = self.metrics['Global']['total_profit'] / self.metrics['Global']['winning_trades']
            if self.metrics['Global']['losing_trades'] > 0:
                self.metrics['Global']['avg_loss'] = self.metrics['Global']['total_loss'] / self.metrics['Global']['losing_trades']
            self.metrics['Global']['last_updated'] = datetime.now()

            # --- DRAWDOWN CALCULATION (Remains Global) ---
            logger.info("Fetching account history for drawdown calculation...")
            balances = self.mt5_handler.get_account_history(days=30) # Increased lookback
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
                
                self.metrics["Global"]["current_drawdown"] = current_drawdown
                self.metrics["Global"]["max_drawdown"] = max_drawdown
            
            # Log a summary
            for name, stats in self.metrics.items():
                logger.info(f"📊 Performance Metrics [{name}]: "
                            f"Trades={stats['total_trades']}, Win Rate={stats['win_rate']:.2%}, "
                            f"P/F={stats['profit_factor']:.2f}")
            
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
                self.metrics["Global"]["avg_signal_quality"] = total_quality / trades_with_quality
                
                self.metrics["Global"]["high_quality_trades"] = quality_metrics["high"]["total"]
                self.metrics["Global"]["medium_quality_trades"] = quality_metrics["medium"]["total"]
                self.metrics["Global"]["low_quality_trades"] = quality_metrics["low"]["total"]
                
                self.metrics["Global"]["high_quality_win_rate"] = quality_metrics["high"]["wins"] / quality_metrics["high"]["total"] if quality_metrics["high"]["total"] > 0 else 0.0
                self.metrics["Global"]["medium_quality_win_rate"] = quality_metrics["medium"]["wins"] / quality_metrics["medium"]["total"] if quality_metrics["medium"]["total"] > 0 else 0.0
                self.metrics["Global"]["low_quality_win_rate"] = quality_metrics["low"]["wins"] / quality_metrics["low"]["total"] if quality_metrics["low"]["total"] > 0 else 0.0
            
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
                "average_quality": self.metrics["Global"]["avg_signal_quality"] * 100,  # Convert to percentage
                "high_quality": {
                    "trades": self.metrics["Global"]["high_quality_trades"],
                    "win_rate": self.metrics["Global"]["high_quality_win_rate"] * 100
                },
                "medium_quality": {
                    "trades": self.metrics["Global"]["medium_quality_trades"],
                    "win_rate": self.metrics["Global"]["medium_quality_win_rate"] * 100
                },
                "low_quality": {
                    "trades": self.metrics["Global"]["low_quality_trades"],
                    "win_rate": self.metrics["Global"]["low_quality_win_rate"] * 100
                }
            },
            "generated_at": datetime.now()
        }
        
        return report 