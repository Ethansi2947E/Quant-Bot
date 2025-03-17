# Advanced Forex Trading Bot Platform

A sophisticated algorithmic trading system that implements Smart Money Concepts (SMC), multi-timeframe analysis, order block detection, and advanced risk management for automated forex trading via MetaTrader 5.

## 🌟 Features

- **Smart Money Concepts (SMC)** - Implementation of professional institutional trading patterns
- **Multi-Timeframe Analysis** - Coherent analysis across different timeframes for robust signals
- **Advanced Order Block Detection** - Identification of high-probability trading zones
- **Risk Management System** - Dynamic position sizing and sophisticated risk controls
- **Real-time Trading** - Direct integration with MetaTrader 5 for execution
- **Telegram Notifications** - Instant trade alerts and performance updates
- **Modern Dashboard** - Real-time performance monitoring and analytics
- **Backtesting Capabilities** - Test strategies against historical data

## 📊 System Architecture

The system is organized into specialized modules, each handling a specific aspect of the trading process:

### Core Components

- **Trading Bot** - Central orchestrator that manages the trading workflow
- **Signal Generators** - Implements different trading strategies and pattern detection
- **MT5 Handler** - Interface with MetaTrader 5 for market data and execution
- **Risk Manager** - Position sizing and risk control
- **Telegram Bot** - Communication and alerts

### Analysis Modules

- **Market Analysis** - Trend detection and market structure identification
- **SMC Analysis** - Smart Money Concepts pattern detection
- **MTF Analysis** - Multi-timeframe correlation and alignment
- **Volume Analysis** - Order flow and volume pattern analysis
- **Divergence Analysis** - RSI and other oscillator divergences
- **POI Detector** - Points of Interest (order blocks, fair value gaps)

## ⚙️ Setup and Installation

### Prerequisites

- Python 3.10+
- MetaTrader 5 terminal installed and configured
- Telegram Bot Token (for alerts)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Ethansi2947E/Trading-Bot2.git
   cd Trading-Bot2
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows
   .\.venv\Scripts\activate
   # Linux/Mac
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration:
   ```
   MT5_USERNAME=your_username
   MT5_PASSWORD=your_password
   MT5_SERVER=your_broker_server
   TELEGRAM_TOKEN=your_telegram_bot_token
   ADMIN_CHAT_ID=your_chat_id
   ```

## 🚀 Usage

### Starting the Trading Bot

Run the main trading bot:

```bash
python main.py
```

### Starting the Dashboard

Run the dashboard for monitoring:

```bash
python start-dashboard.py
```

Access the dashboard at http://localhost:3000

### Configuration

The system is highly configurable through various config files located in the `config/` directory:

- **Trading Parameters** - Risk levels, symbols, timeframes
- **Signal Generator Settings** - Strategy-specific parameters
- **Dashboard Configuration** - UI and reporting options

## 📈 Trading Strategies

The system implements several sophisticated trading strategies:

### 1. SH+BMS+RTO Strategy

Combines Stop Hunt (SH), Break of Market Structure (BMS), and Return to Origin (RTO) patterns for high-probability entries.

### 2. Turtle Soup Strategy

A counter-trend strategy that capitalizes on failed breakouts and price manipulation.

### 3. AMD Strategy

Accumulation, Manipulation, Distribution pattern recognition for trend continuation.

## 📊 Dashboard

The modern Next.js dashboard provides:

- Real-time trade tracking
- Performance analytics
- Equity curve visualization
- Win/loss statistics
- Detailed trade history
- Signal monitoring

## 📝 Documentation

Detailed documentation is available in the `docs/` directory:

- [Trading Bot](docs/trading_bot.md) - Core system architecture
- [Signal Generators](docs/signal_generator1.md) - Trading strategies
- [MT5 Handler](docs/mt5_handler.md) - MetaTrader 5 integration
- [Market Analysis](docs/market_analysis.md) - Technical analysis tools
- [Risk Management](docs/risk_manager.md) - Position sizing and risk controls
- [Telegram Integration](docs/telegram_bot.md) - Alerts and notifications

## 🛠 Development and Extending

### Adding New Strategies

Create a new signal generator by extending the base class:

```python
from src.signal_generator1 import SignalGenerator

class MyCustomStrategy(SignalGenerator):
    async def generate_signals(self, symbol, timeframe):
        # Implement your strategy logic
        pass
```

### Customizing Risk Management

Extend the risk manager to implement custom risk rules:

```python
from src.risk_manager import RiskManager

class EnhancedRiskManager(RiskManager):
    def validate_trade(self, trade_params):
        # Add custom validation logic
        return super().validate_trade(trade_params)
```

## 📋 Project Structure

```
├── config/               # Configuration files
├── data/                 # Data storage
├── docs/                 # Documentation
├── logs/                 # Log files
├── src/                  # Source code
│   ├── dashboard_api.py  # Dashboard backend API
│   ├── database.py       # Database operations
│   ├── divergence_analysis.py  # Divergence detection
│   ├── market_analysis.py  # Market structure analysis
│   ├── mt5_handler.py    # MetaTrader 5 interface
│   ├── mtf_analysis.py   # Multi-timeframe analysis
│   ├── poi_detector.py   # Points of Interest detector
│   ├── risk_manager.py   # Risk management
│   ├── signal_generator1.py  # Strategy implementation
│   ├── smc_analysis.py   # Smart Money Concepts
│   ├── telegram_bot.py   # Telegram integration
│   ├── trading_bot.py    # Main trading bot
│   └── volume_analysis.py  # Volume analysis
├── trading-dash/         # Next.js dashboard
├── backtest.py           # Backtesting script
├── main.py               # Main entry point
├── requirements.txt      # Python dependencies
└── start-dashboard.py    # Dashboard starter
```

## 📌 Best Practices

- **Risk Management First**: Always prioritize risk controls and proper position sizing
- **Strategy Validation**: Test strategies thoroughly in backtesting before live trading
- **Monitoring**: Regularly review logs and dashboard metrics
- **Updates**: Keep MetaTrader 5 and all dependencies updated
- **Backup**: Maintain regular backups of trading data and configurations

## 🔒 Security

- Sensitive credentials are stored in `.env` file (not in git repository)
- Telegram bot uses secure authentication
- Trading parameters are validated before execution
- Error alerts for unusual system behavior

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MetaTrader 5 API
- Python-Telegram-Bot
- Next.js and React for the dashboard
- Pandas and NumPy for data analysis 