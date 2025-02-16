import { NextResponse } from 'next/server';

export async function GET() {
  try {
    // TODO: Replace this with actual data from your trading bot's database
    const mockPerformanceData = {
      totalReturn: 15.75,
      statistics: {
        totalTrades: 150,
        profitableTrades: 90,
        losingTrades: 60,
        winRate: 60,
        averageProfitPerTrade: 125.50,
        averageLossPerTrade: -75.30,
        largestWin: 500.00,
        largestLoss: -250.00,
        profitFactor: 1.67,
        averageTradeDuration: 120,
      },
      riskMetrics: {
        valueAtRisk: 2.5,
        expectedShortfall: 3.2,
        beta: 0.85,
        volatility: 12.5,
        sharpeRatio: 1.8,
        maxDrawdown: 8.5,
        riskRewardRatio: 1.5,
      },
      strategies: [
        {
          strategyId: "trend-following",
          strategyName: "Trend Following",
          returns: 18.5,
          trades: 75,
          winRate: 65,
          sharpeRatio: 2.1,
          maxDrawdown: 7.2,
        },
        {
          strategyId: "mean-reversion",
          strategyName: "Mean Reversion",
          returns: 12.8,
          trades: 45,
          winRate: 58,
          sharpeRatio: 1.6,
          maxDrawdown: 9.1,
        },
        {
          strategyId: "breakout",
          strategyName: "Breakout",
          returns: 15.2,
          trades: 30,
          winRate: 55,
          sharpeRatio: 1.7,
          maxDrawdown: 8.8,
        },
      ],
      costs: {
        totalTransactionCosts: 1250.00,
        averageSlippage: 0.15,
        totalCommissions: 750.00,
        costPerTrade: 8.33,
      },
      equityCurve: Array.from({ length: 30 }, (_, i) => ({
        timestamp: new Date(2024, 0, i + 1).toISOString(),
        equity: 10000 * (1 + Math.sin(i / 10) * 0.1 + i / 100),
        drawdown: Math.max(0, Math.sin(i / 8) * 5),
      })),
      monthlyReturns: Array.from({ length: 12 }, (_, i) => ({
        year: 2024,
        month: i + 1,
        return: (Math.random() * 10 - 2).toFixed(2),
      })),
      tradeDistribution: {
        byTimeOfDay: {
          "00:00-04:00": 15,
          "04:00-08:00": 25,
          "08:00-12:00": 45,
          "12:00-16:00": 35,
          "16:00-20:00": 20,
          "20:00-24:00": 10,
        },
        byDayOfWeek: {
          "Monday": 35,
          "Tuesday": 32,
          "Wednesday": 28,
          "Thursday": 30,
          "Friday": 25,
        },
        byAssetType: {
          "Crypto": 60,
          "Forex": 50,
          "Stocks": 40,
        },
      },
    };

    return NextResponse.json(mockPerformanceData);
  } catch (error) {
    console.error('Error fetching performance data:', error);
    return NextResponse.json(
      { error: 'Failed to fetch performance data' },
      { status: 500 }
    );
  }
} 