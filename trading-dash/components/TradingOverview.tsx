"use client"

import type React from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart
} from "recharts"
import { TrendingUp, ArrowUpCircle, ArrowDownCircle, BarChart2 } from "lucide-react"

interface TradingOverviewProps {
  data: any
}

const TradingOverview: React.FC<TradingOverviewProps> = ({ data }) => {
  if (!data) return null

  return (
    <Card className="bg-white/5 backdrop-blur-xl border-white/10 rounded-xl overflow-hidden">
      <CardHeader className="border-b border-white/10">
        <div className="flex items-center justify-between">
        <div>
            <CardTitle className="text-xl font-semibold text-white">Profit History</CardTitle>
            <p className="text-sm text-white/60 mt-1">Performance over time</p>
        </div>
          <div className="flex items-center gap-2">
            <button className="px-3 py-1 rounded-lg bg-white/5 hover:bg-white/10 text-white/60 hover:text-white text-sm transition-colors">
              24H
            </button>
            <button className="px-3 py-1 rounded-lg bg-white/5 hover:bg-white/10 text-white/60 hover:text-white text-sm transition-colors">
              7D
            </button>
            <button className="px-3 py-1 rounded-lg bg-white/10 text-white text-sm">
              30D
            </button>
            <button className="px-3 py-1 rounded-lg bg-white/5 hover:bg-white/10 text-white/60 hover:text-white text-sm transition-colors">
              ALL
            </button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="p-6">
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data.profit_history}>
              <defs>
                <linearGradient id="profitGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="rgb(59, 130, 246)" stopOpacity={0.5} />
                  <stop offset="100%" stopColor="rgb(59, 130, 246)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid 
                strokeDasharray="3 3" 
                stroke="rgba(255,255,255,0.1)" 
                horizontal={true}
                vertical={false}
              />
              <XAxis 
                dataKey="timestamp" 
                stroke="rgba(255,255,255,0.5)"
                tickFormatter={(value) => {
                  const date = new Date(value);
                  return `${date.getHours()}:${date.getMinutes().toString().padStart(2, '0')}`;
                }}
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
                tickFormatter={(value) => `$${value.toLocaleString()}`}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: 'rgba(17,17,17,0.9)', 
                  border: '1px solid rgba(255,255,255,0.2)',
                  borderRadius: '8px',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                  color: 'white',
                  padding: '12px'
                }}
                labelStyle={{ color: 'rgba(255,255,255,0.7)', marginBottom: '4px' }}
                labelFormatter={(value) => {
                  const date = new Date(value);
                  return date.toLocaleString();
                }}
                formatter={(value: any) => [`$${value.toLocaleString()}`, 'Profit']}
              />
              <Area
                type="monotone"
                dataKey="profit"
                stroke="#3b82f6"
                strokeWidth={2}
                fill="url(#profitGradient)"
                animationDuration={500}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-6">
          <div className="bg-white/5 rounded-lg p-3">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-lg bg-blue-500/10 flex items-center justify-center">
                <TrendingUp className="h-4 w-4 text-blue-500" />
              </div>
              <div>
                <p className="text-xs text-white/60">Daily Change</p>
                <p className="text-sm font-semibold text-white mt-0.5">
                  {((data.profit_history[data.profit_history.length - 1]?.profit || 0) - 
                    (data.profit_history[0]?.profit || 0)).toLocaleString('en-US', {
                    style: 'currency',
                    currency: 'USD'
                  })}
                </p>
              </div>
            </div>
          </div>
          <div className="bg-white/5 rounded-lg p-3">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-lg bg-green-500/10 flex items-center justify-center">
                <ArrowUpCircle className="h-4 w-4 text-green-500" />
              </div>
              <div>
                <p className="text-xs text-white/60">Peak Balance</p>
                <p className="text-sm font-semibold text-white mt-0.5">
                  {Math.max(...data.profit_history.map((p: any) => p.profit)).toLocaleString('en-US', {
                    style: 'currency',
                    currency: 'USD'
                  })}
                </p>
              </div>
            </div>
          </div>
          <div className="bg-white/5 rounded-lg p-3">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-lg bg-red-500/10 flex items-center justify-center">
                <ArrowDownCircle className="h-4 w-4 text-red-500" />
              </div>
              <div>
                <p className="text-xs text-white/60">Lowest Balance</p>
                <p className="text-sm font-semibold text-white mt-0.5">
                  {Math.min(...data.profit_history.map((p: any) => p.profit)).toLocaleString('en-US', {
                    style: 'currency',
                    currency: 'USD'
                  })}
                </p>
              </div>
            </div>
          </div>
          <div className="bg-white/5 rounded-lg p-3">
            <div className="flex items-center gap-2">
              <div className="h-8 w-8 rounded-lg bg-purple-500/10 flex items-center justify-center">
                <BarChart2 className="h-4 w-4 text-purple-500" />
              </div>
              <div>
                <p className="text-xs text-white/60">Average Daily Profit</p>
                <p className="text-sm font-semibold text-white mt-0.5">
                  {(data.profit_history.reduce((acc: number, curr: any) => acc + curr.profit, 0) / 
                    data.profit_history.length).toLocaleString('en-US', {
                    style: 'currency',
                    currency: 'USD'
                  })}
                </p>
              </div>
        </div>
      </div>
      </div>
      </CardContent>
    </Card>
  )
}

export default TradingOverview

