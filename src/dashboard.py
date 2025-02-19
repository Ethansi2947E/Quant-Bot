from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from datetime import datetime, timedelta, time, UTC
from loguru import logger
from collections import deque
import MetaTrader5 as mt5
import os
import numpy as np
import json
import asyncio
from asgiref.wsgi import WsgiToAsgi
from hypercorn.config import Config as HyperConfig
from hypercorn.asyncio import serve
import sys  # add import at top if not present
from pathlib import Path
from fastapi import HTTPException

BASE_DIR = Path(__file__).resolve().parent.parent

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle special types."""
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)

app = Flask(__name__, static_folder='templates/build/static')
app.json_encoder = CustomJSONEncoder

# Configure CORS with more specific settings
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

# Add error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({"error": "Resource not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Convert WSGI app to ASGI
asgi_app = WsgiToAsgi(app)

trading_bot = None
recent_signals = deque(maxlen=50)  # Store last 50 signals
profit_history = deque(maxlen=1000)  # Store last 1000 profit entries

def init_app(bot):
    """Initialize the dashboard with shutdown support."""
    global trading_bot
    trading_bot = bot
    
    # Configure Hypercorn
    config = HyperConfig()
    config.bind = ["0.0.0.0:5000"]  # Allow external connections
    config.use_reloader = False
    
    # Create a shutdown event and start the server with a shutdown trigger
    shutdown_event = asyncio.Event()
    
    async def start_server():
        try:
            await serve(asgi_app, config, shutdown_trigger=shutdown_event.wait)
        except asyncio.CancelledError:
            logger.info("Dashboard server cancelled")
        except Exception as e:
            logger.error(f"Error in dashboard server: {str(e)}")
        finally:
            logger.info("Dashboard server stopped")
            
    server_task = asyncio.create_task(start_server())
    
    # Store shutdown helpers on app
    app.shutdown_event = shutdown_event
    app.server_task = server_task
    
    return app

def set_bot(bot_ref):
    """Set the bot reference for the dashboard."""
    global trading_bot
    trading_bot = bot_ref
    logger.info("Dashboard bot reference updated")

def add_profit_entry(profit_amount: float, trade_type: str = None):
    """Add a new profit entry to the history."""
    try:
        entry = {
            'timestamp': datetime.now(),
            'profit': profit_amount,
            'trade_type': trade_type,
            'cumulative': calculate_cumulative_profit()
        }
        profit_history.append(entry)
        logger.debug(f"Added profit entry: {entry}")
    except Exception as e:
        logger.error(f"Error adding profit entry: {str(e)}")

def calculate_cumulative_profit():
    """Calculate cumulative profit from history."""
    try:
        if not profit_history:
            return 0.0
        return sum(entry['profit'] for entry in profit_history)
    except Exception as e:
        logger.error(f"Error calculating cumulative profit: {str(e)}")
        return 0.0

def get_profit_history(timeframe="24H"):
    """Get profit history for the specified timeframe."""
    try:
        if not profit_history:
            return []
            
        now = datetime.now()
        
        # Convert timeframe to timedelta
        if timeframe == "24H":
            delta = timedelta(hours=24)
        elif timeframe == "7D":
            delta = timedelta(days=7)
        elif timeframe == "30D":
            delta = timedelta(days=30)
        else:  # ALL
            delta = timedelta(days=365)  # Return all data up to a year
            
        cutoff_time = now - delta
        
        # Filter and format profit history
        filtered_history = [
            {
                'timestamp': entry['timestamp'].isoformat(),
                'profit': entry['profit'],
                'trade_type': entry['trade_type'],
                'cumulative': entry['cumulative']
            }
            for entry in profit_history
            if entry['timestamp'] >= cutoff_time
        ]
        
        return filtered_history
    except Exception as e:
        logger.error(f"Error getting profit history: {str(e)}")
        return []

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
        if trading_bot and hasattr(trading_bot, 'market_analysis'):
            # If market_analysis has a callable get_active_pois, use it
            if callable(getattr(trading_bot.market_analysis, 'get_active_pois', None)):
                return trading_bot.market_analysis.get_active_pois()
            # Fallback: use support_resistance_levels if available
            if hasattr(trading_bot.market_analysis, 'support_resistance_levels'):
                return [
                    {
                        'price': level['price'],
                        'type': level['type'],
                        'strength': level.get('strength', 1)
                    }
                    for level in trading_bot.market_analysis.support_resistance_levels
                ]
            # Fallback: use key_levels if available
            elif hasattr(trading_bot.market_analysis, 'key_levels'):
                return trading_bot.market_analysis.key_levels
        return []
    except Exception as e:
        logger.error(f"Error getting active POIs: {str(e)}")
        return []

def get_market_data():
    """Get current market data."""
    try:
        if trading_bot:
            try:
                return trading_bot.get_market_data()
            except AttributeError:
                pass
            try:
                return trading_bot.current_market_data
            except AttributeError:
                pass
            if hasattr(trading_bot, 'market_analysis') and hasattr(trading_bot.market_analysis, 'current_data'):
                return trading_bot.market_analysis.current_data
        # Return an empty dict as fallback
        return {}
    except Exception as e:
        logger.error(f"Error getting market data: {str(e)}")
        return {}

@app.route('/')
def serve_react_app():
    """Serve the React application."""
    return send_from_directory('templates/build', 'index.html')

@app.route('/api/market-analysis')
def get_market_analysis():
    """Get market analysis data for the dashboard."""
    try:
        if not trading_bot or not hasattr(trading_bot, 'market_analysis'):
            return jsonify({
                'error': 'Trading bot or market analysis not initialized'
            }), 500

        # Get current symbol and timeframe with better fallbacks
        symbol = (
            trading_bot.current_symbol if hasattr(trading_bot, 'current_symbol') else
            trading_bot.symbol if hasattr(trading_bot, 'symbol') else
            trading_bot.trading_config.get('symbol', 'Unknown') if hasattr(trading_bot, 'trading_config') else
            'Unknown'
        )
        
        timeframe = (
            trading_bot.timeframe if hasattr(trading_bot, 'timeframe') else
            trading_bot.trading_config.get('timeframe', 'Unknown') if hasattr(trading_bot, 'trading_config') else
            'Unknown'
        )

        # Get market data with better error handling
        market_data = {}
        try:
            if hasattr(trading_bot.market_analysis, 'analyze'):
                current_data = get_market_data()
                if current_data:  # Only analyze if we have data
                    market_data = trading_bot.market_analysis.analyze(current_data, symbol, timeframe)
            elif hasattr(trading_bot.market_analysis, 'current_analysis'):
                market_data = trading_bot.market_analysis.current_analysis
        except Exception as e:
            logger.warning(f"Error during market analysis: {str(e)}")
            # Continue with empty market_data

        # Get session info with fallbacks
        session_info = (
            trading_bot.market_analysis.current_session if hasattr(trading_bot.market_analysis, 'current_session')
            else market_data.get('session_conditions', {})
        )

        # Format response with better fallbacks
        response = {
            'symbol': symbol,
            'timeframe': timeframe,
            'structure_type': market_data.get('structure_type', 'Unknown'),
            'market_bias': (
                market_data.get('trend') or 
                market_data.get('bias') or 
                market_data.get('market_bias', 'neutral')
            ),
            'volume_analysis': {
                'trend': market_data.get('volume_analysis', {}).get('trend', 'Unknown'),
                'strength': market_data.get('volume_analysis', {}).get('strength', 0),
                'average_volume': market_data.get('volume_analysis', {}).get('average_volume', 0),
                'recent_volume': market_data.get('volume_analysis', {}).get('recent_volume', 0),
            },
            'session_conditions': {
                'session': session_info.get('session', 'Unknown'),
                'volatility_factor': session_info.get('volatility_factor', 0),
                'suitable_for_trading': session_info.get('suitable_for_trading', False),
            },
            'key_levels': market_data.get('key_levels', get_active_pois()),
            'quality_score': market_data.get('quality_score', 0),
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error getting market analysis: {str(e)}")
        # Return a more informative default response
        return jsonify({
            'error': str(e),
            'symbol': 'Unknown',
            'timeframe': 'Unknown',
            'structure_type': 'Unknown',
            'market_bias': 'neutral',
            'volume_analysis': {
                'trend': 'Unknown',
                'strength': 0,
                'average_volume': 0,
                'recent_volume': 0
            },
            'session_conditions': {
                'session': 'Unknown',
                'volatility_factor': 0,
                'suitable_for_trading': False
            },
            'key_levels': [],
            'quality_score': 0
        }), 200  # Return 200 instead of 500 to prevent frontend errors

@app.route('/api/trading-status')
def get_trading_status():
    """Get current trading status."""
    try:
        # Initialize MT5 if not already initialized
        if not mt5.initialize():
            logger.error("Failed to initialize MT5")
            return jsonify({'error': 'MT5 not initialized'}), 500

        # Get account info
        account_info = mt5.account_info()
        if not account_info:
            logger.error("Failed to get MT5 account info")
            return jsonify({'error': 'Failed to get account info'}), 500

        # Get all open positions
        positions = mt5.positions_get()
        active_trades = []
        
        if positions:
            for pos in positions:
                # Calculate current profit in percentage
                profit_percent = (pos.profit / account_info.balance) * 100 \
                    if account_info.balance > 0 else 0
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
                    'time': pos.time,
                    'comment': pos.comment
                })

        # Calculate performance metrics
        total_profit = account_info.profit
        equity = account_info.equity
        balance = account_info.balance
        
        # Calculate win rate from closed positions history
        today = datetime.now().date()
        start_of_day = int(datetime.combine(today, time.min).timestamp())
        
        # Get today's deals
        deals = mt5.history_deals_get(start_of_day)
        if deals:
            winning_trades = len([deal for deal in deals if deal.profit > 0])
            total_trades = len(deals)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        else:
            win_rate = 0
            
        # Calculate max drawdown
        deals_with_equity = mt5.history_deals_get(start_of_day)
        if deals_with_equity:
            running_equity = [balance]
            for deal in deals_with_equity:
                running_equity.append(running_equity[-1] + deal.profit)
            peak = max(running_equity)
            max_drawdown = ((peak - min(running_equity)) / peak * 100) if peak > 0 else 0
        else:
            max_drawdown = 0

        return jsonify({
            'mt5_status': "Connected",
            'trading_status': "Enabled" if trading_bot and trading_bot.running else "Disabled",
            'account_info': {
                'balance': balance,
                'equity': equity,
                'profit': total_profit,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'currency': account_info.currency
            },
            'performance': {
                'total_return': ((equity - balance) / balance * 100) if balance > 0 else 0,
                'win_rate': round(win_rate, 2),
                'max_drawdown': round(max_drawdown, 2),
                'sharpe_ratio': 0.0  # This would need historical data to calculate properly
            },
            'active_trades': active_trades,
            'last_update': datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"Error getting trading status: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard_data():
    """Get all dashboard data in a single request."""
    try:
        if not trading_bot:
            return jsonify({
                "status": "error",
                "error": "Trading bot not initialized"
            })
        
        # Get market analysis data
        market_analysis_response = get_market_analysis()
        if isinstance(market_analysis_response, tuple):
            market_analysis = market_analysis_response[0].get_json()
        else:
            market_analysis = market_analysis_response.get_json()
        
        # Get trading status data
        trading_status_response = get_trading_status()
        if isinstance(trading_status_response, tuple):
            trading_status = trading_status_response[0].get_json()
        else:
            trading_status = trading_status_response.get_json()
        
        # Convert any numpy types to Python native types
        def convert_numpy_types(data):
            if isinstance(data, dict):
                return {k: convert_numpy_types(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [convert_numpy_types(item) for item in data]
            elif isinstance(data, np.bool_):
                return bool(data)
            elif isinstance(data, np.integer):
                return int(data)
            elif isinstance(data, np.floating):
                return float(data)
            elif isinstance(data, np.ndarray):
                return data.tolist()
            return data

        # Convert numpy types in the response data
        market_analysis = convert_numpy_types(market_analysis)
        trading_status = convert_numpy_types(trading_status)
        
        # Calculate daily profit
        daily_profit = sum(entry['profit'] for entry in get_profit_history("24H"))
        
        return jsonify({
            "status": "success",
            "data": {
                "market_analysis": market_analysis,
                "trading_status": trading_status,
                "total_profit": calculate_cumulative_profit(),
                "daily_profit": daily_profit,
                "win_rate": trading_status.get("win_rate", 0),
                "total_trades": trading_status.get("total_trades", 0),
                "active_trades": trading_status.get("active_trades", []),
                "trading_status": trading_status.get("trading_status", "Disabled"),
                "profit_history": get_profit_history("24H"),  # Get last 24 hours by default
                "timestamp": datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/trading-data')
def get_trading_data():
    """Get trading data for the dashboard."""
    if not trading_bot:
        return jsonify({'error': 'Trading bot not initialized'}), 500
        
    try:
        # Get combined trading status
        bot_enabled = getattr(trading_bot, 'running', False)
        telegram_enabled = (
            trading_bot.telegram_bot.trading_enabled 
            if hasattr(trading_bot, 'telegram_bot') 
            else False
        )
        trading_status = "Enabled" if (bot_enabled and telegram_enabled) else "Disabled"
        
        # Get MT5 account info
        mt5_info = {}
        if mt5.initialize():
            account_info = mt5.account_info()
            if account_info is not None:
                mt5_info = {
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'profit': account_info.profit,
                    'margin': account_info.margin,
                    'margin_free': account_info.margin_free,
                    'margin_level': account_info.margin_level,
                    'currency': account_info.currency
                }
        
        # Get active trades from MT5 terminal
        positions = mt5.positions_get()
        active_trades = []
        
        if positions:
            for pos in positions:
                # Calculate current profit in percentage
                profit_percent = (pos.profit / account_info.balance) * 100 \
                    if account_info.balance > 0 else 0
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
                    'time': pos.time,
                    'comment': pos.comment
                })
        
        # Calculate total and daily profit
        total_profit = sum(getattr(trade, 'realized_profit', 0) for trade in getattr(trading_bot, 'trades', []))
        daily_profit = sum(
            getattr(trade, 'realized_profit', 0)
            for trade in getattr(trading_bot, 'trades', [])
            if getattr(trade, 'close_time', None) and trade.close_time.date() == datetime.now().date()
        )
        
        # Calculate win rate
        trades = getattr(trading_bot, 'trades', [])
        closed_trades = [t for t in trades if not getattr(t, 'is_open', False)]
        winning_trades = len([t for t in closed_trades if getattr(t, 'realized_profit', 0) > 0])
        win_rate = (winning_trades / len(closed_trades) * 100) if closed_trades else 0
        
        # Get market analysis data
        market_data = getattr(trading_bot, 'market_data', {})
        market_analysis = {
            'market_bias': market_data.get('bias', 'Unknown'),
            'structure_type': market_data.get('structure', 'Unknown'),
            'key_levels': market_data.get('key_levels', [])
        }
        
        return jsonify({
            'mt5_account': mt5_info,
            'total_profit': total_profit,
            'daily_profit': daily_profit,
            'win_rate': round(win_rate, 2),
            'total_trades': len(trades),
            'trading_status': trading_status,
            'active_trades': active_trades,
            'market_analysis': market_analysis,
            'profit_history': [
                {
                    'timestamp': getattr(trade, 'close_time', None).isoformat() if getattr(trade, 'close_time', None) else None,
                    'profit': getattr(trade, 'realized_profit', 0)
                }
                for trade in trades[-24:] if hasattr(trade, 'realized_profit')  # Last 24 trades
            ]
        })
    except Exception as e:
        logger.error(f"Error getting trading data: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/enable-trading', methods=['POST'])
async def enable_trading():
    """Enable trading from dashboard."""
    try:
        if not trading_bot:
            return jsonify({"status": "error", "message": "Trading bot not initialized"}), 500
            
        success = await trading_bot.enable_trading()
        if success:
            return jsonify({"status": "success", "message": "Trading enabled"})
        else:
            return jsonify({"status": "error", "message": "Failed to enable trading"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/disable-trading', methods=['POST'])
async def disable_trading():
    """Disable trading from dashboard."""
    try:
        if not trading_bot:
            return jsonify({"status": "error", "message": "Trading bot not initialized"}), 500
            
        success = await trading_bot.disable_trading()
        if success:
            return jsonify({"status": "success", "message": "Trading disabled"})
        else:
            return jsonify({"status": "error", "message": "Failed to disable trading"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/trading-data/toggle', methods=['POST'])
async def toggle_trading_data():
    """Toggle trading status and return updated trading data."""
    try:
        if not trading_bot:
            return jsonify({"status": "error", "message": "Trading bot not initialized"}), 500
            
        # Get current combined status
        bot_enabled = getattr(trading_bot, 'running', False)
        telegram_enabled = (
            trading_bot.telegram_bot.trading_enabled 
            if hasattr(trading_bot, 'telegram_bot') 
            else False
        )
        current_status = bot_enabled and telegram_enabled
        
        success = False
        if current_status:
            # If currently enabled (both are true), disable both
            success = await trading_bot.disable_trading()
            if success and hasattr(trading_bot, 'telegram_bot'):
                await trading_bot.telegram_bot.disable_trading_core()
        else:
            # If currently disabled (either is false), enable both
            success = await trading_bot.enable_trading()
            if success and hasattr(trading_bot, 'telegram_bot'):
                await trading_bot.telegram_bot.enable_trading_core()
            
        if success:
            # Get updated trading data
            try:
                # Get MT5 account info
                mt5_info = {}
                if mt5.initialize():
                    account_info = mt5.account_info()
                    if account_info is not None:
                        mt5_info = {
                            'balance': account_info.balance,
                            'equity': account_info.equity,
                            'profit': account_info.profit,
                            'margin': account_info.margin,
                            'margin_free': account_info.margin_free,
                            'margin_level': account_info.margin_level,
                            'currency': account_info.currency
                        }
                
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
                            'time': pos.time,
                            'comment': pos.comment
                        })
                
                # Calculate win rate
                trades = getattr(trading_bot, 'trades', [])
                closed_trades = [t for t in trades if not getattr(t, 'is_open', False)]
                winning_trades = len([t for t in closed_trades if getattr(t, 'realized_profit', 0) > 0])
                win_rate = (winning_trades / len(closed_trades) * 100) if closed_trades else 0
                
                # Get new combined status after toggle
                new_bot_enabled = getattr(trading_bot, 'running', False)
                new_telegram_enabled = (
                    trading_bot.telegram_bot.trading_enabled 
                    if hasattr(trading_bot, 'telegram_bot') 
                    else False
                )
                new_status = "Enabled" if (new_bot_enabled and new_telegram_enabled) else "Disabled"
                
                trading_data = {
                    'mt5_account': mt5_info,
                    'total_profit': sum(getattr(t, 'realized_profit', 0) for t in trades),
                    'daily_profit': sum(
                        getattr(t, 'realized_profit', 0)
                        for t in trades
                        if getattr(t, 'close_time', None) 
                        and t.close_time.date() == datetime.now().date()
                    ),
                    'win_rate': round(win_rate, 2),
                    'total_trades': len(trades),
                    'trading_status': new_status,
                    'active_trades': active_trades,
                    'last_update': datetime.now().isoformat()
                }
                
                return jsonify({
                    "status": "success",
                    "message": f"Trading {'disabled' if current_status else 'enabled'} successfully",
                    "data": trading_data
                })
            except Exception as data_error:
                logger.error(f"Error getting trading data after toggle: {str(data_error)}")
                return jsonify({
                    "status": "success",
                    "message": f"Trading {'disabled' if current_status else 'enabled'} successfully, but failed to fetch updated data",
                    "data": None
                })
        else:
            return jsonify({
                "status": "error", 
                "message": f"Failed to {'disable' if current_status else 'enable'} trading"
            }), 500
            
    except Exception as e:
        logger.error(f"Error toggling trading: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/profit-history')
def get_profit_history_endpoint():
    """Get profit history data endpoint."""
    try:
        timeframe = request.args.get('timeframe', '24H')
        history = get_profit_history(timeframe)
        
        if not history:
            # Return empty data structure instead of empty list
            return jsonify({
                'data': [],
                'cumulative_profit': 0,
                'timeframe': timeframe
            })
            
        return jsonify({
            'data': history,
            'cumulative_profit': calculate_cumulative_profit(),
            'timeframe': timeframe
        })
    except Exception as e:
        logger.error(f"Error in profit history endpoint: {str(e)}")
        return jsonify({'error': 'Failed to fetch profit history'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')