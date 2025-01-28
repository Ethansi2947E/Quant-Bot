# Advanced MT5 Trading Bot

A sophisticated algorithmic trading bot that integrates with MetaTrader 5, featuring advanced technical analysis, risk management, and real-time notifications via Telegram.

## Features

### Core Trading Features
- Multi-timeframe analysis
- Smart Money Concepts (SMC) analysis
- Advanced signal generation
- Risk management system
- Position sizing calculator
- Trade management with trailing stops

### Integration
- MetaTrader 5 connection
- Telegram bot for notifications and control
- Real-time market data processing
- News impact analysis

### Monitoring & Analysis
- Performance metrics tracking
- Trade history logging
- Real-time error monitoring
- Market structure analysis

## Prerequisites

- Python 3.10+
- MetaTrader 5 Terminal
- Telegram Bot Token
- Required Python packages (see requirements.txt)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Configuration

1. MetaTrader 5 Setup:
   - Install MT5 Terminal
   - Configure login credentials in .env
   - Set appropriate trading permissions

2. Telegram Bot Setup:
   - Create bot via BotFather
   - Get bot token
   - Add to .env file
   - Configure allowed user IDs

3. Trading Parameters:
   - Set risk management parameters
   - Configure trading pairs
   - Set timeframes
   - Adjust performance monitoring

## Usage

### Starting the Bot

1. Start the bot:
```bash
python main.py
```

2. Initialize via Telegram:
```
/start - Start the bot
/enable - Enable trading
/help - View all commands
```

### Available Commands

#### Trading Controls
- `/enable` - Enable automated trading
- `/disable` - Disable trading
- `/status` - Check bot status

#### Monitoring
- `/metrics` - View performance metrics
- `/history` - View trade history
- `/risk` - View risk metrics

#### System Controls
- `/start` - Initialize bot
- `/stop` - Stop bot and close trades
- `/help` - Show help message

## Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=src tests/
```

## Project Structure

```
trading_bot/
├── src/                    # Source code
│   ├── market_analysis.py  # Market analysis logic
│   ├── signal_generator.py # Signal generation
│   ├── risk_manager.py    # Risk management
│   ├── mt5_handler.py     # MT5 integration
│   ├── telegram_bot.py    # Telegram integration
│   └── models.py          # Data models
├── tests/                  # Test suite
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── performance/       # Performance tests
├── docs/                   # Documentation
├── config/                 # Configuration files
├── logs/                   # Log files
├── requirements.txt        # Python dependencies
└── .env                   # Environment variables
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- [Core System Architecture](core-system-architecture.md)
- [Signal Generation Logic](signal-generation-logic.md)
- [Risk Management](risk-management.md)
- [Telegram Integration](telegram-integration.md)
- [Testing Guide](testing.md)
- [Configuration Guide](configuration.md)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MetaTrader 5 for providing the trading platform
- python-telegram-bot for Telegram integration
- Technical analysis libraries contributors 