'use client';

import React, { useEffect, useState } from 'react';
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { formatNumber } from '@/lib/utils';
import { Badge } from './ui/badge';
import { ArrowUpCircle, ArrowDownCircle, BarChart2, PieChart } from 'lucide-react';
import type { IPerformanceData, ICostAnalysis } from '../types/performance';

interface PerformanceData {
  total_return: number;
  win_rate: number;
  max_drawdown: number;
  sharpe_ratio: number;
  risk_reward_ratio: number;
  profit_factor: number;
  average_win: number;
  average_loss: number;
  largest_win: number;
  largest_loss: number;
  consecutive_wins: number;
  consecutive_losses: number;
}

interface EquityCurvePoint {
  timestamp: string;
  equity: number;
  drawdown: number;
}

interface TradingData {
  performance: PerformanceData;
  equity_curve: EquityCurvePoint[];
}

export function PerformanceAnalytics() {
  const [data, setData] = useState<TradingData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPerformanceData = async () => {
      try {
        setLoading(true);
        const response = await fetch('/api/trading-status');
        if (!response.ok) {
          throw new Error('Failed to fetch performance data');
        }
        const result = await response.json();
        setData(result);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'An error occurred');
      } finally {
        setLoading(false);
      }
    };

    fetchPerformanceData();
    const interval = setInterval(fetchPerformanceData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <div>Loading performance data...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  if (!data) {
    return <div>No performance data available</div>;
  }

  return (
    <Tabs defaultValue="overview" className="w-full">
      <TabsList>
        <TabsTrigger value="overview">Overview</TabsTrigger>
        <TabsTrigger value="risk">Risk Metrics</TabsTrigger>
        <TabsTrigger value="equity">Equity Curve</TabsTrigger>
      </TabsList>

      <TabsContent value="overview">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Total Return</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatNumber(data.performance.total_return)}%</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Win Rate</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatNumber(data.performance.win_rate)}%</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Profit Factor</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatNumber(data.performance.profit_factor)}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatNumber(data.performance.sharpe_ratio)}</div>
            </CardContent>
          </Card>
        </div>
      </TabsContent>

      <TabsContent value="risk">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Max Drawdown</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatNumber(data.performance.max_drawdown)}%</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Risk/Reward Ratio</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{formatNumber(data.performance.risk_reward_ratio)}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Average Win</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">${formatNumber(data.performance.average_win)}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Average Loss</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">${formatNumber(data.performance.average_loss)}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Largest Win</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">${formatNumber(data.performance.largest_win)}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Largest Loss</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">${formatNumber(data.performance.largest_loss)}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Consecutive Wins</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{data.performance.consecutive_wins}</div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Consecutive Losses</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold">{data.performance.consecutive_losses}</div>
            </CardContent>
          </Card>
        </div>
      </TabsContent>

      <TabsContent value="equity">
        <Card>
          <CardHeader>
            <CardTitle>Equity Curve</CardTitle>
            <CardDescription>Account equity and drawdown over time</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-[400px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart
                  data={data.equity_curve}
                  margin={{
                    top: 5,
                    right: 30,
                    left: 20,
                    bottom: 5,
                  }}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tickFormatter={(timestamp) => new Date(timestamp).toLocaleDateString()}
                  />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip
                    formatter={(value: number, name: string) => [
                      name === 'equity' ? `$${formatNumber(value)}` : `${formatNumber(value)}%`,
                      name === 'equity' ? 'Equity' : 'Drawdown'
                    ]}
                    labelFormatter={(label) => new Date(label).toLocaleString()}
                  />
                  <Line
                    yAxisId="left"
                    type="monotone"
                    dataKey="equity"
                    stroke="#4ade80"
                    name="equity"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="drawdown"
                    stroke="#f43f5e"
                    name="drawdown"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  );
} 