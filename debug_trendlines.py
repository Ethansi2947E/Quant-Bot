"""
Debug script for trendline detection and visualization

This improved version includes:
- Enhanced filtering to produce fewer, higher quality trendlines
- Clustering algorithm to eliminate redundant similar trendlines
- Quality scoring based on r-squared values and number of touches
- Maximum limit of 8 trendlines per chart type (bullish/bearish)
- Prioritization of the most significant trendlines
"""
import asyncio
import os
from datetime import datetime
from pathlib import Path

# Ensure matplotlib uses Agg backend for headless environments
import matplotlib
matplotlib.use('Agg')

from src.mt5_handler import MT5Handler
from src.strategy.breakout_reversal_strategy import BreakoutReversalStrategy
from loguru import logger

async def debug_trendlines():
    """Run trendline detection with visualization enabled"""
    logger.info("Starting trendline debug visualization")
    
    # Create debug plots directory if it doesn't exist
    debug_dir = Path("debug_plots")
    debug_dir.mkdir(exist_ok=True)
    
    # Initialize MT5 connection
    mt5_handler = MT5Handler()
    if not mt5_handler.connected:
        if not mt5_handler.initialize():
            logger.error("Failed to connect to MT5")
            return False
    logger.info("Connected to MT5 successfully")
    
    # Symbol to analyze
    symbols = ["XAGUSD", "ETHUSD", "EURUSD"]
    
    # Choose timeframes
    primary_tf = "M5"
    higher_tf = "H1"
    
    # Initialize the strategy
    strategy = BreakoutReversalStrategy(
        primary_timeframe=primary_tf,
        higher_timeframe=higher_tf,
        mt5_handler=mt5_handler,
    )
    
    # Fetch data for the specified symbols and timeframes
    market_data = {}
    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}")
        market_data[symbol] = {}
        
        # Get primary timeframe data (last 1000 candles)
        primary_data = mt5_handler.get_market_data(symbol, primary_tf, 1000)
        higher_data = mt5_handler.get_market_data(symbol, higher_tf, 200)
        
        if primary_data is not None and higher_data is not None:
            market_data[symbol][primary_tf] = primary_data
            market_data[symbol][higher_tf] = higher_data
            logger.info(f"Data fetched successfully for {symbol} - Primary: {len(primary_data)} candles, Higher: {len(higher_data)} candles")
        else:
            logger.error(f"Failed to fetch data for {symbol}")
    
    if not market_data:
        logger.error("No data available for analysis")
        return False
    
    # Call the strategy with debug_visualize=True to force trendline updates
    logger.info("Calling strategy with debug_visualize=True")
    signals = await strategy.generate_signals(market_data, debug_visualize=True)
    
    logger.info(f"Generated {len(signals)} signals")
    logger.info("Debug visualization complete - check the debug_plots folder")
    
    # Final cleanup
    mt5_handler.shutdown()
    return True

if __name__ == "__main__":
    # Print a clear starting message
    print("\n" + "="*80)
    print("IMPROVED TRENDLINE DETECTION VISUALIZATION TOOL")
    print("="*80)
    print("This script will generate visualizations of trendline detection")
    print("with enhanced filtering, clustering, and quality scoring")
    print("Images will be saved to the 'debug_plots' folder")
    print("="*80 + "\n")
    
    # Run the async function
    asyncio.run(debug_trendlines())
    
    # Print completion message
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("The improved algorithm now identifies fewer, higher-quality trendlines")
    print("with better clustering to remove redundant lines.")
    print("Check the 'debug_plots' folder for the generated images")
    print("="*80 + "\n") 