export interface ITradeStatistics {
  totalTrades: number;
  profitableTrades: number;
  losingTrades: number;
  winRate: number;
  averageProfitPerTrade: number;
  averageLossPerTrade: number;
  largestWin: number;
  largestLoss: number;
  profitFactor: number;
  averageTradeDuration: number;
}

export interface IRiskMetrics {
  valueAtRisk: number;
  expectedShortfall: number;
  beta: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  riskRewardRatio: number;
}

export interface IStrategyPerformance {
  strategyId: string;
  strategyName: string;
  returns: number;
  trades: number;
  winRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
}

export interface ICostAnalysis {
  totalTransactionCosts: number;
  averageSlippage: number;
  totalCommissions: number;
  costPerTrade: number;
}

export interface IPerformanceData {
  totalReturn: number;
  statistics: ITradeStatistics;
  riskMetrics: IRiskMetrics;
  strategies: IStrategyPerformance[];
  costs: ICostAnalysis;
  equityCurve: Array<{
    timestamp: string;
    equity: number;
    drawdown: number;
  }>;
  monthlyReturns: Array<{
    year: number;
    month: number;
    return: number;
  }>;
  tradeDistribution: {
    byTimeOfDay: Record<string, number>;
    byDayOfWeek: Record<string, number>;
    byAssetType: Record<string, number>;
  };
} 