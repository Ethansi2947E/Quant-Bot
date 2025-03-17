"use client"

import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"

// Mock data for performance chart
const performanceData = [
  { date: "2023-05-15", equity: 10000, drawdown: 0 },
  { date: "2023-05-16", equity: 10120, drawdown: 0 },
  { date: "2023-05-17", equity: 10050, drawdown: -70 },
  { date: "2023-05-18", equity: 10200, drawdown: 0 },
  { date: "2023-05-19", equity: 10350, drawdown: 0 },
  { date: "2023-05-20", equity: 10275, drawdown: -75 },
  { date: "2023-05-21", equity: 10400, drawdown: 0 },
  { date: "2023-05-22", equity: 10380, drawdown: -20 },
  { date: "2023-05-23", equity: 10450, drawdown: 0 },
  { date: "2023-05-24", equity: 10600, drawdown: 0 },
  { date: "2023-05-25", equity: 10550, drawdown: -50 },
  { date: "2023-05-26", equity: 10700, drawdown: 0 },
  { date: "2023-05-27", equity: 10850, drawdown: 0 },
  { date: "2023-05-28", equity: 10800, drawdown: -50 },
  { date: "2023-05-29", equity: 10950, drawdown: 0 },
  { date: "2023-05-30", equity: 11100, drawdown: 0 },
  { date: "2023-05-31", equity: 11050, drawdown: -50 },
  { date: "2023-06-01", equity: 11200, drawdown: 0 },
  { date: "2023-06-02", equity: 11350, drawdown: 0 },
  { date: "2023-06-03", equity: 11300, drawdown: -50 },
  { date: "2023-06-04", equity: 11450, drawdown: 0 },
  { date: "2023-06-05", equity: 11600, drawdown: 0 },
  { date: "2023-06-06", equity: 11550, drawdown: -50 },
  { date: "2023-06-07", equity: 11700, drawdown: 0 },
  { date: "2023-06-08", equity: 11850, drawdown: 0 },
  { date: "2023-06-09", equity: 11800, drawdown: -50 },
  { date: "2023-06-10", equity: 11950, drawdown: 0 },
  { date: "2023-06-11", equity: 12100, drawdown: 0 },
  { date: "2023-06-12", equity: 12050, drawdown: -50 },
  { date: "2023-06-13", equity: 12200, drawdown: 0 },
  { date: "2023-06-14", equity: 12350, drawdown: 0 },
  { date: "2023-06-15", equity: 12500, drawdown: 0 },
]

export function PerformanceChart() {
  const formatDate = (dateString: string) => {
    const date = new Date(dateString)
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric" })
  }

  const formatCurrency = (value: number) => {
    return `$${value.toFixed(2)}`
  }

  return (
    <div className="h-[400px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={performanceData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
            </linearGradient>
            <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#ef4444" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="date" tickFormatter={formatDate} tick={{ fontSize: 12 }} tickLine={false} axisLine={false} />
          <YAxis
            tickFormatter={formatCurrency}
            tick={{ fontSize: 12 }}
            tickLine={false}
            axisLine={false}
            domain={["dataMin - 500", "dataMax + 500"]}
          />
          <CartesianGrid strokeDasharray="3 3" vertical={false} />
          <Tooltip
            formatter={(value: number) => [`$${value.toFixed(2)}`, ""]}
            labelFormatter={(label) => formatDate(label)}
          />
          <Area
            type="monotone"
            dataKey="equity"
            stroke="#10b981"
            fillOpacity={1}
            fill="url(#colorEquity)"
            strokeWidth={2}
            activeDot={{ r: 6 }}
            name="Equity"
          />
          <Area
            type="monotone"
            dataKey="drawdown"
            stroke="#ef4444"
            fillOpacity={1}
            fill="url(#colorDrawdown)"
            strokeWidth={2}
            name="Drawdown"
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

