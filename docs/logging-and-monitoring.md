# Logging and Monitoring

## Overview
The trading bot will implement a comprehensive logging and monitoring system to track its activities, performance, and errors. A simple dashboard will provide real-time insights, and critical alerts will be sent via Telegram.

---

## Logging Structure

The trading bot implements comprehensive logging to track all important decisions, actions, and events. Logs are written to both the console and log files in the `logs` directory.

### Log Levels

1. **INFO**: General information about bot operation
   - Bot startup and shutdown
   - Connection status
   - Trading session changes
   - Symbol analysis start/end

2. **DEBUG**: Detailed information about bot decisions
   - Market structure analysis results
   - POI detection details
   - Signal generation process
   - Risk calculations

3. **WARNING**: Important events that require attention
   - Connection issues
   - Trade execution delays
   - Configuration warnings

4. **ERROR**: Critical issues that affect bot operation
   - Connection failures
   - Trade execution failures
   - System errors

### What Gets Logged

#### 1. Market Analysis
```
[TIME] | INFO | Starting analysis for SYMBOL on TIMEFRAME
[TIME] | DEBUG | Market Structure: Trend=DIRECTION, Key Levels=[LEVELS]
[TIME] | DEBUG | POI Detection: Found ORDER_BLOCKS at [PRICES]
[TIME] | DEBUG | Session Status: ASIAN/LONDON/NY, Volatility=VALUE
```

#### 2. Signal Generation
```
[TIME] | DEBUG | Primary Filters: TF_Alignment=RESULT, Session_Check=RESULT
[TIME] | DEBUG | Confirmation Signals: SMT=RESULT, Liquidity=RESULT
[TIME] | INFO | Generated SIGNAL_TYPE signal for SYMBOL
[TIME] | DEBUG | Signal Confidence: VALUE%, Risk-Reward: VALUE
```

#### 3. Trade Management
```
[TIME] | INFO | Entering trade: SYMBOL, Direction=DIRECTION
[TIME] | DEBUG | Position Size: LOTS, Risk=VALUE%
[TIME] | DEBUG | Entry=PRICE, SL=PRICE, TP=PRICE
[TIME] | INFO | Trade executed successfully: ORDER_ID
```

#### 4. Position Monitoring
```
[TIME] | DEBUG | Monitoring trade ORDER_ID: Current P/L=VALUE
[TIME] | DEBUG | Adjusting SL to PRICE based on REASON
[TIME] | INFO | Closing trade ORDER_ID: Result=PROFIT/LOSS
```

### Example Log Output
```
2024-01-22 02:44:40 | INFO | MT5 connection established successfully
2024-01-22 02:44:40 | INFO | Starting trading bot...
2024-01-22 02:44:41 | INFO | Starting analysis for EURUSD on H1
2024-01-22 02:44:41 | DEBUG | Market Structure: Trend=Bullish, Key Levels=[1.0850, 1.0900]
2024-01-22 02:44:42 | DEBUG | POI Detection: Found OB at [1.0855-1.0860]
2024-01-22 02:44:42 | DEBUG | Session Status: London, Volatility=Medium
2024-01-22 02:44:43 | DEBUG | Primary Filters: TF_Alignment=True, Session_Check=True
2024-01-22 02:44:43 | DEBUG | Confirmation Signals: SMT=True, Liquidity=True
2024-01-22 02:44:44 | INFO | Generated BUY signal for EURUSD
2024-01-22 02:44:44 | DEBUG | Signal Confidence: 85%, Risk-Reward: 1:2.5
```

## Real-time Monitoring

### Dashboard (Flask/Dash)
The bot includes a web-based dashboard that displays:

1. **Active Trades**
   - Current positions
   - Entry price, SL, TP
   - Current P/L
   - Trade duration

2. **Market Analysis**
   - Current market structure
   - Detected POIs
   - Recent signals

3. **Performance Metrics**
   - Win rate
   - Average RR
   - Daily/Weekly P/L
   - Drawdown

4. **System Status**
   - Connection status
   - CPU/Memory usage
   - Error rates
   - Last update time

### Accessing the Dashboard
1. The dashboard runs on `http://localhost:5000` by default
2. Access metrics and charts through the web interface
3. Real-time updates via WebSocket connection

### Telegram Notifications
Important events are also sent via Telegram:
1. Trade signals and executions
2. Position updates
3. Risk warnings
4. System status changes

## Configuration

### Logging Configuration
```python
LOG_CONFIG = {
    "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
    "level": "DEBUG",  # Set to INFO, DEBUG, WARNING, or ERROR
    "rotation": "1 day",
    "retention": "1 month",
    "compression": "zip",
}
```

### Dashboard Configuration
```python
DASHBOARD_CONFIG = {
    "host": "localhost",
    "port": 5000,
    "debug": True,
    "update_interval": 5,  # seconds
}
```

## Best Practices

1. Keep DEBUG level enabled during testing and initial deployment
2. Monitor log files regularly for warnings and errors
3. Set up log rotation to manage disk space
4. Use the dashboard for real-time monitoring
5. Configure Telegram alerts for critical events
6. Archive logs periodically for analysis

## Example Workflow

### Logging
- Bot logs trade execution details to `bot.log`.
- Example: `[2023-10-15 10:30:00] [INFO] [Trade Execution] Trade executed: EURUSD Buy at 1.1200`

### Monitoring
- User accesses the dashboard to view performance metrics and trade history.

### Alerting
- Bot sends a Telegram alert for a failed trade.
- Example: "Critical Error: Failed to execute trade - Insufficient margin"

## Technologies and Libraries
- **Logging**: Loguru for logging.
- **Dashboard**: Dash for creating the monitoring dashboard.
- **Alerting**: python-telegram-bot for sending alerts.