from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from datetime import datetime, timedelta
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
CORS(app)  # Enable CORS for all routes

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
    config.bind = ["localhost:5000"]
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
        now = datetime.now()
        if timeframe == "24H":
            cutoff = now - timedelta(hours=24)
        elif timeframe == "7D":
            cutoff = now - timedelta(days=7)
        elif timeframe == "30D":
            cutoff = now - timedelta(days=30)
        else:  # ALL
            return list(profit_history)
        
        return [entry for entry in profit_history if entry['timestamp'] >= cutoff]
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
        mt5_status = "Connected" if mt5.initialize() else "Disconnected"
        trading_status = "Enabled" if trading_bot and trading_bot.running else "Disabled"
        
        active_symbols = trading_bot.trading_config["symbols"] if trading_bot else ["EURUSD", "GBPUSD", "USDJPY"]
        win_rate = calculate_win_rate(trading_bot.trades if trading_bot else [])
        total_profit = calculate_total_profit(trading_bot.trades if trading_bot else [])
        total_trades = len(trading_bot.trades) if trading_bot and hasattr(trading_bot, 'trades') else 0
        
        active_trades = []
        if trading_bot and mt5.initialize():
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

        return jsonify({
            'mt5_status': mt5_status,
            'trading_status': trading_status,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_trades': total_trades,
            'active_symbols': active_symbols,
            'active_trades': active_trades,
            'recent_signals': get_recent_signals(),
            'active_pois': get_active_pois(),
            'last_update': datetime.now().isoformat()
        })

    except (KeyError, AttributeError) as e:
        logger.error(f"Error getting trading status: {str(e)}")
        return jsonify({'error': str(e)})

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
        
        # Get data from trading bot
        active_trades = []
        if hasattr(trading_bot, 'trades'):
            active_trades = [
                {
                    'symbol': trade.symbol,
                    'type': trade.type,
                    'entry': trade.entry_price,
                    'sl': trade.stop_loss,
                    'tp': trade.take_profit,
                    'profit': trade.current_profit,
                    'volume': trade.volume if hasattr(trade, 'volume') else 0.01
                }
                for trade in trading_bot.trades if getattr(trade, 'is_open', True)
            ]
        
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

@app.route('/api/toggle-trading', methods=['POST'])
async def toggle_trading():
    """Toggle the trading bot's status."""
    if not trading_bot:
        return jsonify({'error': 'Trading bot not initialized'}), 500
        
    try:
        # Get current states
        bot_enabled = getattr(trading_bot, 'running', False)
        telegram_enabled = (
            trading_bot.telegram_bot.trading_enabled 
            if hasattr(trading_bot, 'telegram_bot') 
            else False
        )
        
        logger.info(f"Current states - Bot: {bot_enabled}, Telegram: {telegram_enabled}")
        
        # Determine desired state (opposite of current state)
        current_state = "Enabled" if telegram_enabled else "Disabled"
        desired_state = "Disabled" if current_state == "Enabled" else "Enabled"
        
        logger.info(f"Toggling trading from {current_state} to {desired_state}")
        
        if desired_state == "Enabled":
            try:
                # Initialize MT5 first if not already initialized
                if not mt5.initialize():
                    logger.error("Failed to initialize MT5 connection")
                    return jsonify({
                        'error': 'MT5 connection failed',
                        'message': 'Please ensure MetaTrader 5 is running and try again'
                    }), 500

                # Test connection by trying to get account info
                account_info = mt5.account_info()
                if account_info is None:
                    mt5.shutdown()
                    logger.error("Failed to get MT5 account info")
                    return jsonify({
                        'error': 'MT5 connection failed',
                        'message': 'Could not get account information. Please check your MT5 connection'
                    }), 500

                # Start the bot if not running
                if not bot_enabled:
                    asyncio.create_task(trading_bot.start())
                # Enable trading in Telegram bot
                if hasattr(trading_bot, 'telegram_bot'):
                    trading_bot.telegram_bot.trading_enabled = True
                status = "Enabled"
                
            except Exception as e:
                logger.error(f"Error enabling trading: {str(e)}")
                return jsonify({
                    'error': 'Failed to enable trading',
                    'message': str(e)
                }), 500
        else:  # Disable
            try:
                # Only disable trading, don't stop the bot
                if hasattr(trading_bot, 'telegram_bot'):
                    trading_bot.telegram_bot.trading_enabled = False
                status = "Disabled"
            except Exception as e:
                logger.error(f"Error disabling trading: {str(e)}")
                return jsonify({
                    'error': 'Failed to disable trading',
                    'message': str(e)
                }), 500

        logger.info(f"Trading status changed to: {status}")
        
        return jsonify({
            'success': True,
            'trading_status': status,
            'message': f'Trading has been {status.lower()}'
        })
        
    except Exception as e:
        logger.error(f"Error toggling trading status: {str(e)}")
        return jsonify({
            'error': str(e),
            'message': 'Failed to toggle trading status'
        }), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')