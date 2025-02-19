/**
 * Represents the trading data received from the API
 */
export interface TradingData {
  /** Current profit/loss in the base currency */
  currentPnL: number
  /** Total number of open positions */
  openPositions: number
  /** Current account balance */
  accountBalance: number
  /** Trading bot status (enabled/disabled) */
  isTradingEnabled: boolean
  /** Last update timestamp */
  lastUpdate: string
  /** Win rate percentage */
  winRate: number
  /** Total number of trades */
  totalTrades: number
  /** List of recent trades */
  recentTrades: Trade[]
  /** Market analysis data */
  marketAnalysis: MarketAnalysis
}

/**
 * Represents a single trade
 */
export interface Trade {
  /** Unique trade identifier */
  id: string
  /** Trading symbol (e.g., "EURUSD") */
  symbol: string
  /** Trade type (buy/sell) */
  type: 'BUY' | 'SELL'
  /** Trade volume/lot size */
  volume: number
  /** Entry price */
  entryPrice: number
  /** Current price or exit price */
  currentPrice: number
  /** Trade profit/loss */
  pnl: number
  /** Trade open timestamp */
  openTime: string
  /** Trade close timestamp (if closed) */
  closeTime?: string
  /** Trade status */
  status: 'OPEN' | 'CLOSED'
}

/**
 * Represents market analysis data
 */
export interface MarketAnalysis {
  /** Market trend direction */
  trend: 'BULLISH' | 'BEARISH' | 'NEUTRAL'
  /** Market volatility level */
  volatility: 'HIGH' | 'MEDIUM' | 'LOW'
  /** Technical indicators */
  indicators: {
    /** Moving Average values */
    movingAverages: {
      ma20: number
      ma50: number
      ma200: number
    }
    /** Relative Strength Index */
    rsi: number
    /** Moving Average Convergence Divergence */
    macd: {
      value: number
      signal: number
      histogram: number
    }
  }
  /** Latest market signals */
  signals: Signal[]
}

/**
 * Represents a trading signal
 */
export interface Signal {
  /** Signal type */
  type: 'ENTRY' | 'EXIT'
  /** Trading direction */
  direction: 'LONG' | 'SHORT'
  /** Signal strength (0-100) */
  strength: number
  /** Signal timestamp */
  timestamp: string
  /** Additional signal information */
  info?: string
}

export interface ProfitHistory {
  data: Array<{
    timestamp: string;
    profit: number;
    trade_type: string | null;
    cumulative: number;
  }>;
  cumulative_profit: number;
  timeframe: string;
} 