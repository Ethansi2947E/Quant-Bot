from flask import Flask, render_template
from datetime import datetime
from loguru import logger
from collections import deque
import MetaTrader5 as mt5

app = Flask(__name__)
bot = None
recent_signals = deque(maxlen=50)  # Store last 50 signals

def init_app(trading_bot=None):
    """Initialize the dashboard."""
    global bot
    bot = trading_bot
    return app

def set_bot(trading_bot):
    """Set the bot reference for the dashboard."""
    global bot
    bot = trading_bot
    logger.info("Dashboard bot reference updated")

def add_signal(signal_data):
    """Add a new signal to the dashboard."""
    try:
        signal_data['timestamp'] = datetime.utcnow()
        recent_signals.append(signal_data)
        logger.debug(f"Added signal to dashboard: {signal_data}")
    except Exception as e:
        logger.error(f"Error adding signal to dashboard: {str(e)}")

def calculate_win_rate(trades):
    """Calculate win rate from trades."""
    if not trades:
        return 0
    winning_trades = sum(1 for trade in trades if trade.profit > 0)
    return round((winning_trades / len(trades)) * 100, 2)

def calculate_total_profit(trades):
    """Calculate total profit from trades."""
    if not trades:
        return 0.0
    return sum(trade.profit for trade in trades)

def get_recent_signals():
    """Get recent trading signals."""
    try:
        return list(recent_signals)
    except Exception as e:
        logger.error(f"Error getting recent signals: {str(e)}")
        return []

def get_active_pois():
    """Get active points of interest."""
    try:
        if not bot or not hasattr(bot, 'active_pois'):
            return []
        return bot.active_pois
    except Exception as e:
        logger.error(f"Error getting active POIs: {str(e)}")
        return []

@app.route('/')
def dashboard():
    """Render the dashboard page."""
    try:
        # Get trading bot status and data
        mt5_status = "Connected" if mt5.initialize() else "Disconnected"
        trading_status = "Enabled" if bot and bot.running else "Disabled"
        
        # Get recent signals
        recent_signals = get_recent_signals()
        
        # Get active symbols from bot config
        active_symbols = bot.trading_config["symbols"] if bot else ["EURUSD", "GBPUSD", "USDJPY"]
        
        # Get trading stats
        win_rate = calculate_win_rate(bot.trades if bot else [])
        total_profit = calculate_total_profit(bot.trades if bot else [])
        total_trades = len(bot.trades) if bot and hasattr(bot, 'trades') else 0
        
        # Get active trades
        active_trades = []
        if bot and mt5.initialize():
            try:
                for symbol in active_symbols:
                    positions = mt5.positions_get(symbol=symbol)
                    if positions:
                        for pos in positions:
                            active_trades.append({
                                'symbol': pos.symbol,
                                'type': 'BUY' if pos.type == 0 else 'SELL',
                                'entry': pos.price_open,
                                'sl': pos.sl,
                                'tp': pos.tp,
                                'profit': pos.profit
                            })
            except Exception as e:
                logger.error(f"Error getting active trades: {str(e)}")
        
        return render_template(
            'dashboard.html',
            mt5_status=mt5_status,
            trading_status=trading_status,
            win_rate=win_rate,
            total_profit=total_profit,
            total_trades=total_trades,
            active_symbols=active_symbols,
            active_trades=active_trades,
            recent_signals=recent_signals,
            active_pois=get_active_pois(),
            last_update=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}")
        return f"Error loading dashboard: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000) 