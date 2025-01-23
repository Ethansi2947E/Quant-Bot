# Getting Started

This guide will walk you through the process of setting up and running the trading bot on your local machine.

## Prerequisites

Before getting started, ensure that you have the following prerequisites installed:

- Python 3.7 or higher
- pip (Python package installer)
- MetaTrader 5 (MT5) platform
- Telegram account (for bot interaction)

## Installation

1. Clone the trading bot repository from GitHub:
   ```
   git clone https://github.com/yourusername/trading-bot.git
   ```

2. Navigate to the project directory:
   ```
   cd trading-bot
   ```

3. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   ```

4. Activate the virtual environment:
   - For Windows:
     ```
     venv\Scripts\activate
     ```
   - For macOS and Linux:
     ```
     source venv/bin/activate
     ```

5. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Create a copy of the `config.example.py` file and rename it to `config.py`:
   ```
   cp config.example.py config.py
   ```

2. Open the `config.py` file in a text editor and update the following configuration sections:
   - `MT5_CONFIG`: Set the MT5 server, login, password, and timeout values.
   - `TRADING_CONFIG`: Specify the trading symbols, timeframes, risk per trade, and maximum daily risk.
   - `TELEGRAM_CONFIG`: Provide the Telegram bot token and allowed user IDs.
   - `AI_CONFIG`: Set the OpenAI API key, news API key, and sentiment threshold (if applicable).
   - `DB_CONFIG`: Specify the database URL and echo settings.
   - `LOG_CONFIG`: Customize the logging format, level, rotation, retention, and compression settings.

3. Save the `config.py` file.

## Running the Bot

1. Ensure that the MT5 platform is running and you are logged in to your trading account.

2. Start the trading bot by running the following command:
   ```
   python main.py
   ```

3. The bot will initialize and start monitoring the specified trading symbols and timeframes.

4. Interact with the bot using the configured Telegram bot. Send commands and receive updates and alerts based on the bot's configuration.

## Monitoring and Logging

- The bot's activity and logs can be monitored through the console output and the log files generated in the `logs` directory.

- The log files are rotated based on the configured rotation interval and retained according to the retention period specified in the `LOG_CONFIG` section of the `config.py` file.

## Troubleshooting

- If you encounter any issues or errors during the setup or running of the bot, refer to the troubleshooting guide in the documentation or seek assistance from the project's support channels.

- Ensure that you have the latest version of the bot and all the required dependencies installed.

- Double-check your configuration settings in the `config.py` file to ensure they are correct and compatible with your environment.

## Conclusion

By following this getting started guide, you should now have the trading bot set up and running on your local machine. Feel free to explore the bot's features, customize its behavior through the configuration file, and monitor its performance using the provided logging and monitoring mechanisms.

Remember to use the bot responsibly and always monitor its activity closely. Happy trading! 