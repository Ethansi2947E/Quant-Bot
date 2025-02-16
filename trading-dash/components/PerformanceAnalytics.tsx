'use client';

import { useEffect, useState } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { ArrowUpCircle, ArrowDownCircle, BarChart2, PieChart } from 'lucide-react';
import type { IPerformanceData } from '../types/performance';

export function PerformanceAnalytics() {
  const [performanceData, setPerformanceData] = useState<IPerformanceData | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const fetchPerformanceData = async () => {
      try {
        // Use the same data source as the main dashboard
        const response = await fetch('/api/trading-data');
        const data = await response.json();
        
        // Transform the data into our performance format
        const transformedData = {
          totalReturn: data.mt5_account?.profit || 0,
          statistics: {
            totalTrades: data.total_trades || 0,
            profitableTrades: Math.floor((data.win_rate / 100) * data.total_trades) || 0,
            losingTrades: data.total_trades - Math.floor((data.win_rate / 100) * data.total_trades) || 0,
            winRate: data.win_rate || 0,
            averageProfitPerTrade: (data.mt5_account?.profit / data.total_trades) || 0,
            averageLossPerTrade: data.average_loss || 0,
            largestWin: data.largest_win || 0,
            largestLoss: data.largest_loss || 0,
            profitFactor: data.profit_factor || 0,
            averageTradeDuration: data.average_trade_duration || 0,
          },
          riskMetrics: {
            valueAtRisk: data.value_at_risk || 0,
            expectedShortfall: data.expected_shortfall || 0,
            beta: data.beta || 0,
            volatility: data.volatility || 0,
            sharpeRatio: data.sharpe_ratio || 0,
            maxDrawdown: data.max_drawdown || 0,
            riskRewardRatio: data.risk_reward_ratio || 0,
          },
          strategies: data.strategies || [],
          costs: {
            totalTransactionCosts: data.transaction_costs || 0,
            averageSlippage: data.average_slippage || 0,
            totalCommissions: data.total_commissions || 0,
            costPerTrade: (data.total_commissions / data.total_trades) || 0,
          },
          equityCurve: data.profit_history?.map((point: any) => ({
            timestamp: point.timestamp,
            equity: point.profit,
            drawdown: point.drawdown || 0,
          })) || [],
          monthlyReturns: data.monthly_returns || [],
          tradeDistribution: {
            byTimeOfDay: data.trade_distribution?.by_time || {},
            byDayOfWeek: data.trade_distribution?.by_day || {},
            byAssetType: data.trade_distribution?.by_asset || {},
          },
        };
        
        setPerformanceData(transformedData);
      } catch (error) {
        console.error('Failed to fetch performance data:', error);
      } finally {
        setIsLoading(false);
      }
    };

    fetchPerformanceData();
    // Refresh every 5 seconds to match the main dashboard
    const interval = setInterval(fetchPerformanceData, 5000);
    return () => clearInterval(interval);
  }, []);

  if (isLoading) {
    return <div className="animate-pulse">Loading performance data...</div>;
  }

  if (!performanceData) {
    return <div className="text-gray-400">No performance data available</div>;
  }

  return (
    <div className="space-y-6">
      {/* Key Performance Indicators */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="bg-white/5 backdrop-blur-xl border-white/10">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-white/80">Total Return</CardTitle>
            <BarChart2 className="h-4 w-4 text-blue-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {performanceData.totalReturn.toFixed(2)}%
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white/5 backdrop-blur-xl border-white/10">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-white/80">Sharpe Ratio</CardTitle>
            <PieChart className="h-4 w-4 text-purple-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {performanceData.riskMetrics.sharpeRatio.toFixed(2)}
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white/5 backdrop-blur-xl border-white/10">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-white/80">Win Rate</CardTitle>
            <ArrowUpCircle className="h-4 w-4 text-green-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {performanceData.statistics.winRate.toFixed(2)}%
            </div>
          </CardContent>
        </Card>

        <Card className="bg-white/5 backdrop-blur-xl border-white/10">
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium text-white/80">Max Drawdown</CardTitle>
            <ArrowDownCircle className="h-4 w-4 text-red-500" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-white">
              {performanceData.riskMetrics.maxDrawdown.toFixed(2)}%
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Detailed Analysis Tabs */}
      <Card className="bg-white/5 backdrop-blur-xl border-white/10 rounded-xl overflow-hidden">
        <CardContent className="p-6">
          <Tabs defaultValue="equity" className="space-y-4">
            <TabsList className="bg-white/5 text-white rounded-lg p-1">
              <TabsTrigger value="equity" className="data-[state=active]:bg-white/10 rounded-md">
                Equity Curve
              </TabsTrigger>
              <TabsTrigger value="statistics" className="data-[state=active]:bg-white/10 rounded-md">
                Statistics
              </TabsTrigger>
              <TabsTrigger value="risk" className="data-[state=active]:bg-white/10 rounded-md">
                Risk Metrics
              </TabsTrigger>
              <TabsTrigger value="strategies" className="data-[state=active]:bg-white/10 rounded-md">
                Strategies
              </TabsTrigger>
            </TabsList>

            <TabsContent value="equity" className="space-y-4">
              <div className="h-[400px] mt-4 rounded-xl overflow-hidden bg-white/5 p-4">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={performanceData.equityCurve}>
                    <CartesianGrid 
                      strokeDasharray="3 3" 
                      stroke="rgba(255,255,255,0.1)" 
                      horizontal={true}
                      vertical={false}
                    />
                    <XAxis 
                      dataKey="timestamp" 
                      stroke="rgba(255,255,255,0.5)"
                      tickFormatter={(value) => new Date(value).toLocaleDateString()}
                      tick={{ fontSize: 12 }}
                      axisLine={false}
                      tickLine={false}
                      dy={10}
                    />
                    <YAxis 
                      stroke="rgba(255,255,255,0.5)"
                      tick={{ fontSize: 12 }}
                      axisLine={false}
                      tickLine={false}
                      dx={-10}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'rgba(17,17,17,0.9)', 
                        border: '1px solid rgba(255,255,255,0.2)',
                        borderRadius: '8px',
                        color: 'white',
                        padding: '12px'
                      }}
                      labelStyle={{ color: 'rgba(255,255,255,0.7)' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="equity"
                      stroke="#3b82f6"
                      strokeWidth={2}
                      dot={false}
                      name="Equity"
                    />
                    <Line
                      type="monotone"
                      dataKey="drawdown"
                      stroke="#ef4444"
                      strokeWidth={2}
                      dot={false}
                      name="Drawdown"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="statistics" className="space-y-4 text-white">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-white/60">Total Trades</span>
                    <span>{performanceData.statistics.totalTrades}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">Profitable Trades</span>
                    <span>{performanceData.statistics.profitableTrades}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">Average Profit</span>
                    <span>${performanceData.statistics.averageProfitPerTrade.toFixed(2)}</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-white/60">Profit Factor</span>
                    <span>{performanceData.statistics.profitFactor.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">Largest Win</span>
                    <span>${performanceData.statistics.largestWin.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">Largest Loss</span>
                    <span>${performanceData.statistics.largestLoss.toFixed(2)}</span>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="risk" className="space-y-4 text-white">
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-white/60">Value at Risk</span>
                    <span>{performanceData.riskMetrics.valueAtRisk.toFixed(2)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">Beta</span>
                    <span>{performanceData.riskMetrics.beta.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">Volatility</span>
                    <span>{performanceData.riskMetrics.volatility.toFixed(2)}%</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-white/60">Risk/Reward</span>
                    <span>{performanceData.riskMetrics.riskRewardRatio.toFixed(2)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-white/60">Expected Shortfall</span>
                    <span>{performanceData.riskMetrics.expectedShortfall.toFixed(2)}%</span>
                  </div>
                </div>
              </div>
            </TabsContent>

            <TabsContent value="strategies" className="space-y-4">
              <div className="space-y-4">
                {performanceData.strategies.map((strategy) => (
                  <div
                    key={strategy.strategyId}
                    className="flex items-center justify-between border-b border-white/10 pb-2"
                  >
                    <div>
                      <h4 className="font-medium text-white">{strategy.strategyName}</h4>
                      <div className="flex space-x-2 mt-1">
                        <Badge variant="outline">
                          {strategy.trades} trades
                        </Badge>
                        <Badge variant="outline">
                          {strategy.winRate.toFixed(2)}% win rate
                        </Badge>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-white">
                        {strategy.returns.toFixed(2)}%
                      </div>
                      <div className="text-sm text-white/60">
                        Sharpe: {strategy.sharpeRatio.toFixed(2)}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </TabsContent>
          </Tabs>
        </CardContent>
      </Card>
    </div>
  );
} 