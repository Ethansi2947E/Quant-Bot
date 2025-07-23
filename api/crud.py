from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from typing import Optional
from . import models
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

# Mapping for timeframe string to timedelta
TIMEFRAME_DELTAS = {
    "1W": timedelta(weeks=1),
    "1M": timedelta(days=30),
    "3M": timedelta(days=90),
    "6M": timedelta(days=180),
    "1Y": timedelta(days=365),
    "ALL": None # Special case for all data
}

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

def get_available_assets(db: Session):
    """
    Returns a list of unique asset symbols from the trades table.
    """
    return [result[0] for result in db.query(models.Trade.symbol).distinct().all()]

def _get_filtered_trades(db: Session, filters: dict):
    """
    Central function to get trades based on asset, date range, and timeframe.
    """
    query = db.query(models.Trade)
    
    asset = filters.get("asset")
    if asset and asset != "All Assets":
        query = query.filter(models.Trade.symbol == asset)
        
    start_date_str = filters.get("start_date")
    if start_date_str:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        query = query.filter(models.Trade.close_time >= start_date)

    end_date_str = filters.get("end_date")
    if end_date_str:
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
        query = query.filter(models.Trade.close_time <= end_date)
        
    # Timeframe is handled separately in the functions that use it.
    return query.order_by(models.Trade.close_time.asc()).all()


def get_performance_chart_data(db: Session, filters: dict):
    """
    Generates the equity curve and drawdown data for the performance chart,
    filtered by the specified timeframe.
    """
    timeframe = filters.get("timeframe", "1Y")
    delta = TIMEFRAME_DELTAS.get(timeframe.upper())
    
    # Base query for all trades that could be relevant
    base_query = db.query(models.Trade)
    asset = filters.get("asset")
    if asset and asset != "All Assets":
        base_query = base_query.filter(models.Trade.symbol == asset)
    
    start_date_filter = None
    if delta:
        start_date_filter = datetime.utcnow() - delta

    # Filter trades for the chart period
    period_query = base_query
    if start_date_filter:
        period_query = period_query.filter(models.Trade.close_time >= start_date_filter)

    trades = period_query.order_by(models.Trade.close_time.asc()).all()
    
    initial_balance = 10000
    
    starting_equity = initial_balance
    if start_date_filter:
        profit_before_window = db.query(func.sum(models.Trade.profit))\
            .filter(models.Trade.symbol == asset if asset and asset != "All Assets" else True)\
            .filter(models.Trade.close_time < start_date_filter).scalar() or 0
        starting_equity += profit_before_window

    equity_curve = []
    current_equity = starting_equity
    peak_equity = starting_equity
    max_drawdown_value = 0
    
    # Add an initial point for the chart
    first_date = trades[0].close_time if trades else datetime.utcnow()
    start_point_date = start_date_filter if delta else (first_date - timedelta(seconds=1))
    equity_curve.append({
        "date": start_point_date.strftime('%Y-%m-%d %H:%M:%S'),
        "equity": starting_equity,
        "drawdown": 0,
    })

    for trade in trades:
        current_equity += trade.profit
        
        if current_equity > peak_equity:
            peak_equity = current_equity
        
        drawdown_value = peak_equity - current_equity
        if drawdown_value > max_drawdown_value:
            max_drawdown_value = drawdown_value

        current_drawdown_pct = (drawdown_value / peak_equity) * 100 if peak_equity > 0 else 0

        equity_curve.append({
            "date": trade.close_time.strftime('%Y-%m-%d %H:%M:%S'),
            "equity": current_equity,
            "drawdown": -current_drawdown_pct,
        })
        
    max_drawdown_pct = (max_drawdown_value / peak_equity) * 100 if peak_equity > 0 else 0

    # If the dataset is large, resample to daily points to improve frontend performance.
    # A threshold of 366 points is chosen to allow daily data for up to a year without resampling.
    if len(equity_curve) > 366:
        resampled_points = {}
        # The first point is always the initial balance, so we preserve it.
        # It might not have a trade on its "day".
        initial_point = equity_curve[0]
        
        # Group subsequent points by day, keeping only the last one for each day.
        for point in equity_curve[1:]:
            day = point["date"][:10] # Extract YYYY-MM-DD
            resampled_points[day] = point
            
        # Combine the initial point with the sorted daily points.
        final_curve = [initial_point] + [resampled_points[day] for day in sorted(resampled_points.keys())]
        
        return {
            "series": final_curve,
            "max_drawdown_percent": max_drawdown_pct
        }

    return {
        "series": equity_curve,
        "max_drawdown_percent": max_drawdown_pct
    }

def get_performance_overview(db: Session, filters: dict):
    """
    Calculates and returns the data for the 'Overview' tab on the performance page.
    """
    trades = _get_filtered_trades(db, filters)
    performance_chart_result = get_performance_chart_data(db, filters=filters)
    
    if not trades:
        # Create a list of placeholder metrics if no trades are found
        titles = ["Total Return", "Annualized Return", "Sharpe Ratio", "Max Drawdown", "Calmar Ratio", "Volatility", "Beta", "Alpha"]
        metrics = [{"title": title, "value": "N/A", "change": "", "color": "text-gray-500"} for title in titles]
        return {"metrics": metrics, "performanceChart": []}

    initial_balance = 10000
    
    # --- Daily Returns Calculation ---
    daily_profits = defaultdict(float)
    for trade in trades:
        day_key = trade.close_time.date()
        daily_profits[day_key] += trade.profit
    
    daily_returns = [p / initial_balance for p in daily_profits.values()] if daily_profits else []

    # --- Metric Calculations ---
    total_return = sum(daily_returns)
    
    annualized_volatility = np.std(daily_returns) * np.sqrt(252) if len(daily_returns) > 1 else 0
    
    num_days = (max(daily_profits.keys()) - min(daily_profits.keys())).days if daily_profits else 0
    num_years = num_days / 365.25 if num_days > 0 else 0
    annualized_return = ((1 + total_return) ** (1 / num_years)) - 1 if num_years > 0 and total_return > -1 else 0
    
    mean_daily_return = np.mean(daily_returns) if daily_returns else 0
    sharpe_ratio = (mean_daily_return * 252) / annualized_volatility if annualized_volatility > 0 else 0

    max_drawdown_pct = performance_chart_result.get('max_drawdown_percent', 0)
    calmar_ratio = (annualized_return * 100) / max_drawdown_pct if max_drawdown_pct > 0 else 0

    def get_color(value):
        return "text-green-500" if value >= 0 else "text-red-500"

    metrics = [
        {"title": "Total Return", "value": f"{(total_return * 100):.2f}%", "change": "", "color": get_color(total_return)},
        {"title": "Annualized Return", "value": f"{(annualized_return * 100):.2f}%", "change": "", "color": get_color(annualized_return)},
        {"title": "Sharpe Ratio", "value": f"{sharpe_ratio:.2f}", "change": "", "color": get_color(sharpe_ratio)},
        {"title": "Max Drawdown", "value": f"-{max_drawdown_pct:.2f}%", "change": "", "color": "text-red-500"},
        {"title": "Calmar Ratio", "value": f"{calmar_ratio:.2f}", "change": "", "color": get_color(calmar_ratio)},
        {"title": "Volatility", "value": f"{(annualized_volatility * 100):.2f}%", "change": "", "color": "text-gray-500"},
        {"title": "Beta", "value": "N/A", "change": "", "color": "text-gray-500"},
        {"title": "Alpha", "value": "N/A", "change": "", "color": "text-gray-500"},
    ]
    
    return {
        "metrics": metrics,
        "performanceChart": performance_chart_result["series"]
    }


def get_risk_analysis(db: Session, filters: dict):
    """
    Calculates and returns the data for the 'Risk Analysis' tab.
    """
    trades = _get_filtered_trades(db, filters)
    if not trades:
        return {
            "metrics": [],
            "drawdownChart": [],
            "volatilityChart": []
        }

    initial_balance = 10000
    
    # --- Equity and Drawdown Calculation ---
    equity_curve = [initial_balance]
    peak_equity = initial_balance
    max_drawdown_value = 0
    drawdown_percentages_sq = []
    daily_profits = defaultdict(float)

    for trade in trades:
        # For VaR and ES calculation
        day_key = trade.close_time.strftime('%Y-%m-%d')
        daily_profits[day_key] += trade.profit
        
        # For other metrics
        current_equity = equity_curve[-1] + trade.profit
        equity_curve.append(current_equity)
        
        if current_equity > peak_equity:
            peak_equity = current_equity
        
        drawdown_value = peak_equity - current_equity
        if drawdown_value > max_drawdown_value:
            max_drawdown_value = drawdown_value
            
        drawdown_pct_sq = ((drawdown_value / peak_equity) ** 2) if peak_equity > 0 else 0
        drawdown_percentages_sq.append(drawdown_pct_sq)

    # --- Metric Calculations ---
    profits = [t.profit for t in trades]
    total_profit = sum(profits)

    # 1. Historical VaR and Expected Shortfall (95% confidence)
    daily_profit_values = list(daily_profits.values())
    var_95 = 0
    expected_shortfall_95 = 0
    if daily_profit_values:
        var_95 = np.percentile(daily_profit_values, 5)
        # Expected Shortfall is the average of returns that are worse than VaR
        losses_beyond_var = [loss for loss in daily_profit_values if loss < var_95]
        if losses_beyond_var:
            expected_shortfall_95 = np.mean(losses_beyond_var)

    # 2. Maximum Consecutive Losses
    max_consecutive_losses = 0
    current_consecutive_losses = 0
    for profit in profits:
        if profit < 0:
            current_consecutive_losses += 1
        else:
            if current_consecutive_losses > max_consecutive_losses:
                max_consecutive_losses = current_consecutive_losses
            current_consecutive_losses = 0
    if current_consecutive_losses > max_consecutive_losses:
        max_consecutive_losses = current_consecutive_losses
    
    # 3. Recovery Factor
    recovery_factor = total_profit / max_drawdown_value if max_drawdown_value > 0 else 0

    # 4. Ulcer Index
    ulcer_index = np.sqrt(np.mean(drawdown_percentages_sq)) if drawdown_percentages_sq else 0

    # 5. Sterling Ratio (simplified)
    total_return_pct = (total_profit / initial_balance) * 100 if initial_balance > 0 else 0
    max_drawdown_pct = (max_drawdown_value / peak_equity) * 100 if peak_equity > 0 else 0
    sterling_ratio = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0

    # --- Chart Data Calculations (already using filtered trades) ---
    # 3. Monthly Drawdown Analysis
    monthly_drawdowns = defaultdict(list)
    for trade in trades:
        month_key = trade.close_time.strftime('%Y-%m')
        monthly_drawdowns[month_key].append(trade.profit)

    drawdown_chart_data = []
    for month, month_profits in sorted(monthly_drawdowns.items()):
        month_name = datetime.strptime(month, '%Y-%m').strftime('%b %Y')
        
        month_peak_equity = 0
        month_current_equity = 0
        month_max_dd_value = 0
        for profit in month_profits:
            month_current_equity += profit
            if month_current_equity > month_peak_equity:
                month_peak_equity = month_current_equity
            drawdown = month_peak_equity - month_current_equity
            if drawdown > month_max_dd_value:
                month_max_dd_value = drawdown
        
        monthly_dd_pct = (month_max_dd_value / initial_balance) * 100 if initial_balance > 0 else 0
        
        drawdown_chart_data.append({ "period": month_name, "drawdown": -monthly_dd_pct })

    # 4. Rolling Volatility
    weekly_returns = defaultdict(list)
    for trade in trades:
        week_key = trade.close_time.strftime('%Y-%U')
        weekly_returns[week_key].append(trade.profit)

    volatility_chart_data = []
    for week, week_profits in sorted(weekly_returns.items())[-4:]:
        if len(week_profits) > 1:
            week_num = datetime.strptime(week + '-1', '%Y-%U-%w').isocalendar()[1]
            volatility = np.std(week_profits) 
            volatility_chart_data.append({ "period": f"Week {week_num}", "volatility": volatility })

    metrics = [
        { "metric": "Value at Risk (95%)", "value": f"${var_95:,.2f}", "description": "5th percentile of daily returns (Historical)" },
        { "metric": "Expected Shortfall (95%)", "value": f"${expected_shortfall_95:,.2f}", "description": "Avg loss on days worse than VaR (Historical)" },
        { "metric": "Maximum Consecutive Losses", "value": max_consecutive_losses, "description": "Longest losing streak from filtered trades" },
        { "metric": "Recovery Factor", "value": f"{recovery_factor:.2f}", "description": "Net profit / Max drawdown" },
        { "metric": "Ulcer Index", "value": f"{ulcer_index:.2f}", "description": "Measures depth and duration of drawdown" },
        { "metric": "Sterling Ratio", "value": f"{sterling_ratio:.2f}", "description": "Return / Max Drawdown (simplified)" }
    ]
    
    return {
        "metrics": metrics,
        "drawdownChart": drawdown_chart_data,
        "volatilityChart": volatility_chart_data
    }

def get_monthly_breakdown(db: Session, filters: dict):
    """
    Calculates and returns the data for the 'Monthly Breakdown' tab.
    """
    trades = _get_filtered_trades(db, filters)
    if not trades:
        return { "yearlyStats": [], "monthlyData": [] }

    # Group trades by month
    monthly_data_agg = defaultdict(lambda: {
        'profits': [],
        'trades': 0,
        'winning_trades': 0,
        'daily_profits': defaultdict(float)
    })

    initial_balance = 10000
    
    for trade in trades:
        month_key = trade.close_time.strftime('%Y-%m')
        day_key = trade.close_time.strftime('%Y-%m-%d')
        
        agg = monthly_data_agg[month_key]
        agg['profits'].append(trade.profit)
        agg['trades'] += 1
        if trade.profit > 0:
            agg['winning_trades'] += 1
        agg['daily_profits'][day_key] += trade.profit

    # Process aggregated data into the final format
    monthly_data_final = []
    monthly_returns = []

    for month, data in sorted(monthly_data_agg.items()):
        month_name = datetime.strptime(month, '%Y-%m').strftime('%B %Y')
        total_profit = sum(data['profits'])
        
        return_pct = (total_profit / initial_balance) * 100 if initial_balance > 0 else 0
        monthly_returns.append({"month": month_name, "return": return_pct})

        win_rate = (data['winning_trades'] / data['trades']) * 100 if data['trades'] > 0 else 0
        
        daily_pnl_pct = {day: (profit / initial_balance) * 100 for day, profit in data['daily_profits'].items()}
        best_day_pct = max(daily_pnl_pct.values()) if daily_pnl_pct else 0
        worst_day_pct = min(daily_pnl_pct.values()) if daily_pnl_pct else 0

        monthly_data_final.append({
            "month": month_name,
            "return": round(return_pct, 2),
            "trades": data['trades'],
            "winRate": round(win_rate, 2),
            "bestDay": round(best_day_pct, 2),
            "worstDay": round(worst_day_pct, 2)
        })

    # Calculate Yearly Stats
    if not monthly_returns:
        return { "yearlyStats": [], "monthlyData": [] }

    best_month = max(monthly_returns, key=lambda x: x['return'])
    worst_month = min(monthly_returns, key=lambda x: x['return'])
    positive_months = sum(1 for r in monthly_returns if r['return'] > 0)
    positive_months_pct = (positive_months / len(monthly_returns)) * 100
    
    yearly_stats = [
        { "metric": "Best Month", "value": f"{best_month['month']} (+{best_month['return']:.2f}%)" },
        { "metric": "Worst Month", "value": f"{worst_month['month']} ({worst_month['return']:.2f}%)" },
        { "metric": "Positive Months", "value": f"{positive_months} out of {len(monthly_returns)} ({positive_months_pct:.1f}%)" },
        { "metric": "Average Monthly Return", "value": f"{np.mean([r['return'] for r in monthly_returns]):.2f}%" },
        { "metric": "Monthly Volatility", "value": f"{np.std([r['return'] for r in monthly_returns]):.2f}%" },
        { "metric": "Consistency Score", "value": "N/A" },
    ]
    
    return {
        "yearlyStats": yearly_stats,
        "monthlyData": sorted(monthly_data_final, key=lambda x: datetime.strptime(x['month'], '%B %Y'), reverse=True)
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