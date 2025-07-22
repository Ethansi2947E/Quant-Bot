from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import Optional
from . import models, schemas
from datetime import datetime, timedelta

# This file will contain functions to read/write from the database.

def get_trade_history(db: Session, skip: int = 0, limit: int = 20, sort_by: str = "close_time", sort_order: str = "desc", search: Optional[str] = None):
    """
    Retrieves a paginated, sorted, and searchable list of trade history.
    """
    query = db.query(models.Trade)

    # Apply search filter if provided
    if search:
        search_term = f"%{search}%"
        query = query.filter(models.Trade.symbol.like(search_term))

    # Apply sorting
    sort_column = getattr(models.Trade, sort_by, models.Trade.close_time)
    if sort_order == "desc":
        query = query.order_by(desc(sort_column))
    else:
        query = query.order_by(sort_column)

    total_trades = query.count()
    trades = query.offset(skip).limit(limit).all()
    
    return {
        "trades": trades,
        "total_trades": total_trades
    }

def get_dashboard_kpis(db: Session):
    """
    Calculates and returns the key performance indicators for the main dashboard.
    """
    total_profit_loss = db.query(func.sum(models.Trade.profit)).scalar() or 0.0
    
    total_trades = db.query(func.count(models.Trade.id)).scalar() or 0
    
    winning_trades = db.query(func.count(models.Trade.id)).filter(models.Trade.profit > 0).scalar() or 0
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    # Placeholder for portfolio value - in a real scenario, this might come from account info
    # For now, we can estimate it based on an initial balance + profits.
    initial_balance = 10000 # Example initial balance
    portfolio_value = initial_balance + total_profit_loss

    return {
        "totalProfitLoss": {"value": total_profit_loss, "change": 0.125}, # change is placeholder
        "winRate": {"value": win_rate, "change": 0.021}, # change is placeholder
        "totalTrades": {"value": total_trades, "change": 18}, # change is placeholder
        "portfolioValue": {"value": portfolio_value, "change": 1234.56, "changePercentage": 0.053} # change is placeholder
    }

def get_overview_chart_data(db: Session):
    """
    Gets the data for the weekly profit/loss bar chart on the dashboard.
    """
    last_7_days = datetime.utcnow() - timedelta(days=7)
    
    daily_profits = db.query(
        func.strftime('%Y-%m-%d', models.Trade.close_time).label('day'),
        func.sum(models.Trade.profit).label('total_profit')
    ).filter(models.Trade.close_time >= last_7_days).group_by('day').order_by('day').all()

    # Create a dictionary for quick lookup
    profit_map = {res.day: res.total_profit for res in daily_profits}
    
    # Format for the chart, ensuring all 7 days are present
    chart_data = []
    for i in range(7):
        date = datetime.utcnow().date() - timedelta(days=6-i)
        day_str = date.strftime('%a') # e.g., 'Mon'
        date_key = date.strftime('%Y-%m-%d')
        chart_data.append({
            "name": day_str,
            "total": profit_map.get(date_key, 0)
        })

    return chart_data

def get_performance_chart_data(db: Session, timeframe: str = "1Y"):
    """
    Generates the equity curve and drawdown data for the performance chart.
    """
    # This is a complex calculation. For now, we'll create a simplified version.
    # A full implementation would require tracking balance over time.
    
    trades = db.query(models.Trade).order_by(models.Trade.close_time.asc()).all()
    
    equity_curve = []
    current_equity = 10000 # Example initial balance
    
    for trade in trades:
        current_equity += trade.profit
        equity_curve.append({
            "date": trade.close_time.strftime('%Y-%m-%d'),
            "equity": current_equity,
            "drawdown": 0, # Placeholder for drawdown
            "volume": trade.volume
        })
        
    # This is a simplified return; the actual implementation would need to handle
    # different timeframes (1W, 1M, etc.) by resampling the data.
    return {
        "1W": equity_curve[-7:] if len(equity_curve) > 7 else equity_curve,
        "1M": equity_curve[-30:] if len(equity_curve) > 30 else equity_curve,
        "3M": equity_curve[-90:] if len(equity_curve) > 90 else equity_curve,
        "6M": equity_curve[-180:] if len(equity_curve) > 180 else equity_curve,
        "1Y": equity_curve
    }

def get_performance_overview(db: Session):
    """
    Calculates and returns the data for the 'Overview' tab on the performance page.
    """
    trades = db.query(models.Trade).all()
    if not trades:
        return { "metrics": [], "performanceChart": { "1Y": [] } }

    total_return_pct = (sum(t.profit for t in trades) / 10000) * 100 # Assuming 10k initial balance
    
    # NOTE: These are simplified placeholders. Real calculations are much more complex.
    metrics = [
        { "title": "Total Return", "value": f"{total_return_pct:.2f}%", "change": "+2.1%" },
        { "title": "Annualized Return", "value": "+18.7%", "change": "+1.5%" },
        { "title": "Sharpe Ratio", "value": "1.85", "change": "+0.12" },
        { "title": "Max Drawdown", "value": "-8.2%", "change": "-1.1%" },
        { "title": "Calmar Ratio", "value": "2.28", "change": "+0.15" },
        { "title": "Volatility", "value": "12.4%", "change": "-0.8%" },
        { "title": "Beta", "value": "0.73", "change": "-0.05" },
        { "title": "Alpha", "value": "4.2%", "change": "+0.3%" }
    ]
    
    performance_chart_data = get_performance_chart_data(db) # Reuse the equity curve logic

    return {
        "metrics": metrics,
        "performanceChart": performance_chart_data
    }

def get_risk_analysis(db: Session):
    """
    Calculates and returns the data for the 'Risk Analysis' tab.
    """
    # NOTE: These are simplified placeholders.
    metrics = [
        { "metric": "Value at Risk (95%)", "value": "-$1,250", "description": "Maximum expected loss over 1 day" },
        { "metric": "Expected Shortfall", "value": "-$1,850", "description": "Average loss beyond VaR" },
        { "metric": "Maximum Consecutive Losses", "value": 4, "description": "Longest losing streak" },
        { "metric": "Recovery Factor", "value": 2.87, "description": "Net profit / Max drawdown" },
        { "metric": "Ulcer Index", "value": 3.2, "description": "Measure of downside risk" },
        { "metric": "Sterling Ratio", "value": 1.95, "description": "Return / Average drawdown" }
    ]
    
    drawdown_chart = [
        { "period": "Jan", "drawdown": -2.1 }, { "period": "Feb", "drawdown": -1.5 },
        { "period": "Mar", "drawdown": -4.2 }, { "period": "Apr", "drawdown": -0.8 },
        { "period": "May", "drawdown": -3.1 }, { "period": "Jun", "drawdown": -8.2 },
    ]
    
    volatility_chart = [
        { "period": "Week 1", "volatility": 8.5 }, { "period": "Week 2", "volatility": 12.3 },
        { "period": "Week 3", "volatility": 15.7 }, { "period": "Week 4", "volatility": 9.2 },
    ]

    return {
        "metrics": metrics,
        "drawdownChart": drawdown_chart,
        "volatilityChart": volatility_chart
    }

def get_monthly_breakdown(db: Session):
    """
    Calculates and returns the data for the 'Monthly Breakdown' tab.
    """
    # NOTE: These are simplified placeholders.
    yearly_stats = [
        { "metric": "Best Month", "value": "June 2024 (+8.4%)" },
        { "metric": "Worst Month", "value": "March 2024 (-1.5%)" },
        { "metric": "Positive Months", "value": "5 out of 6 (83.3%)" },
    ]
    
    monthly_data = [
        { "month": "January 2024", "return": 4.2, "trades": 28, "winRate": 71.4, "bestDay": 2.1, "worstDay": -1.3 },
        { "month": "February 2024", "return": 2.8, "trades": 24, "winRate": 66.7, "bestDay": 1.8, "worstDay": -0.9 },
        { "month": "March 2024", "return": -1.5, "trades": 31, "winRate": 58.1, "bestDay": 1.5, "worstDay": -2.8 },
    ]

    return {
        "yearlyStats": yearly_stats,
        "monthlyData": monthly_data
    }

def get_win_loss_analysis(db: Session):
    """
    Calculates and returns all data required for the Win/Loss Analysis page.
    """
    trades = db.query(models.Trade).order_by(models.Trade.close_time.asc()).all()
    if not trades:
        # Return a default structure if there's no data
        return {
            "overview": { "winLossDistribution": [], "keyStatistics": {}, "streakAnalysis": [] },
            "trends": { "winRateOverTime": [], "monthlyDistribution": [] },
            "assetPerformance": [],
            "timingAnalysis": { "byHour": [], "bestHours": [], "mostActiveHours": [] }
        }

    # Overview Tab
    winning_trades = []
    for t in trades:
        if t.profit is not None and t.profit > 0:
            winning_trades.append(t)
            
    losing_trades = []
    for t in trades:
        if t.profit is not None and t.profit <= 0:
            losing_trades.append(t)
            
    win_loss_distribution = [
        { "name": "Winning Trades", "value": len(winning_trades) },
        { "name": "Losing Trades", "value": len(losing_trades) }
    ]
    total_profit = sum(t.profit for t in winning_trades if t.profit is not None)
    total_loss = abs(sum(t.profit for t in losing_trades if t.profit is not None))
    
    key_statistics = {
        "winRate": (len(winning_trades) / len(trades)) * 100 if trades else 0,
        "lossRate": (len(losing_trades) / len(trades)) * 100 if trades else 0,
        "profitFactor": total_profit / total_loss if total_loss > 0 else 0,
        "averageWin": total_profit / len(winning_trades) if winning_trades else 0,
        "averageLoss": total_loss / len(losing_trades) if losing_trades else 0,
    }
    
    # Placeholder for streak analysis as it's complex
    streak_analysis = [
        { "type": "Current Streak", "value": "N/A" },
        { "type": "Longest Win Streak", "value": "N/A" },
        { "type": "Longest Loss Streak", "value": "N/A" },
    ]

    # For other tabs, we will use placeholder data for now
    # A full implementation would require more complex queries (e.g., grouping by month, asset, hour)
    
    return {
        "overview": {
            "winLossDistribution": win_loss_distribution,
            "keyStatistics": key_statistics,
            "streakAnalysis": streak_analysis
        },
        "trends": {
            "winRateOverTime": [{ "date": "Jan", "winRate": 72 }, { "date": "Feb", "winRate": 68 }],
            "monthlyDistribution": [{ "date": "Jan", "wins": 18, "losses": 7 }]
        },
        "assetPerformance": [
            { "asset": "BTC/USD", "wins": 45, "losses": 15, "winRate": 75, "avgWin": 2.3, "avgLoss": -1.8 }
        ],
        "timingAnalysis": {
            "byHour": [{ "hour": "08-12", "wins": 42, "losses": 18, "winRate": 70 }],
            "bestHours": [{ "hour": "20-24", "winRate": 90.6, "wins": 29, "losses": 3 }],
            "mostActiveHours": [{ "hour": "08-12", "totalTrades": 60, "winRate": 70 }]
        }
    }

def get_signals(db: Session, skip: int = 0, limit: int = 20):
    """
    Retrieves a paginated list of the most recent trading signals.
    """
    return db.query(models.Signal).order_by(models.Signal.timestamp.desc()).offset(skip).limit(limit).all() 