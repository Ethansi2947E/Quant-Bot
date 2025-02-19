"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  XAxis,
  YAxis,
  CartesianGrid,
  ResponsiveContainer,
  Area,
  AreaChart,
  Legend,
  Tooltip
} from "recharts"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { API_ENDPOINTS } from "@/app/config"
import { Loader2 } from "lucide-react"

interface ProfitHistory {
  timestamp: string
  profit: number
  trade_type: string | null
  cumulative: number
}

interface ProfitHistoryResponse {
  data: ProfitHistory[]
  cumulative_profit: number
  timeframe: string
}

const TradingOverview = () => {
  const [chartData, setChartData] = useState<ProfitHistory[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [timeframe, setTimeframe] = useState("24H")
  const [totalProfit, setTotalProfit] = useState(0)
  const [profitableCount, setProfitableCount] = useState(0)
  const [totalTrades, setTotalTrades] = useState(0)
  const [refreshing, setRefreshing] = useState(false)

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true)
        setRefreshing(true)
        const response = await fetch(`${API_ENDPOINTS.PROFIT_HISTORY}?timeframe=${timeframe}`)

        if (!response.ok) {
          throw new Error(`Failed to fetch data: ${response.statusText}`)
        }

        const result: ProfitHistoryResponse = await response.json()
        console.log('API response:', result)

        // Validate the response structure
        if (!result.data || !Array.isArray(result.data)) {
          console.error('Invalid response structure:', result)
          throw new Error('Invalid profit history data format')
        }

        // Process profit history data
        const processedData = result.data.map((entry: ProfitHistory) => ({
          ...entry,
          profit: Number(entry.profit),
          cumulative: Number(entry.cumulative)
        }))

        // Calculate statistics
        const profitable = processedData.filter(entry => entry.profit > 0).length
        
        setChartData(processedData)
        setTotalProfit(result.cumulative_profit)
        setProfitableCount(profitable)
        setTotalTrades(processedData.length)
        console.log('Total trades:', processedData.length)
        console.log('Profitable trades:', profitable)
        setError(null)
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'An error occurred while fetching profit history'
        setError(errorMessage)
        console.error('Error fetching data:', err)
      } finally {
        setLoading(false)
        setRefreshing(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 5000)
    return () => clearInterval(interval)
  }, [timeframe])

  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <div className="bg-white/10 backdrop-blur-xl p-4 rounded-lg border border-white/20">
          <p className="text-white/80">{new Date(data.timestamp).toLocaleString()}</p>
          <p className="text-white">
            Trade Profit: ${Number(data.profit).toFixed(2)}
          </p>
          <p className="text-emerald-400">
            Cumulative: ${Number(data.cumulative).toFixed(2)}
          </p>
          {data.trade_type && (
            <p className="text-white/60">
              Trade Type: {data.trade_type}
            </p>
          )}
        </div>
      )
    }
    return null
  }

  if (loading) {
    return (
      <Card className="bg-white/5 backdrop-blur-xl border-white/10 rounded-xl overflow-hidden">
        <CardContent className="p-6">
          <div className="h-[400px] flex items-center justify-center">
            <p className="text-white/60">Loading profit history...</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className="bg-white/5 backdrop-blur-xl border-white/10 rounded-xl overflow-hidden">
        <CardContent className="p-6">
          <div className="h-[400px] flex items-center justify-center">
            <p className="text-white/60">{error}</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="bg-white/5 backdrop-blur-xl border-white/10 rounded-xl overflow-hidden">
      <CardHeader className="border-b border-white/10 p-6">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="text-xl font-semibold text-white flex items-center">
              Profit History
              {refreshing && (
                <div className="relative group">
                  <Loader2 className="animate-spin h-4 w-4 ml-2 text-white/50" />
                  <div className="absolute hidden group-hover:block left-0 mt-2 px-2 py-1 bg-black/80 text-white text-xs rounded">
                    Refreshing data...
                  </div>
                </div>
              )}
            </CardTitle>
            <div className="mt-2 space-x-4">
              <span className="text-white/60">
                Total Profit: <span className={`font-medium ${totalProfit >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                  ${totalProfit.toFixed(2)}
                </span>
              </span>
              <span className="text-white/60">
                Win Rate: <span className="text-emerald-400 font-medium">
                  {totalTrades > 0 ? ((profitableCount / totalTrades) * 100).toFixed(1) : 0}%
                </span>
              </span>
              <span className="text-white/60">
                Total Trades: <span className="text-white font-medium">{totalTrades}</span>
              </span>
            </div>
          </div>
          <Select value={timeframe} onValueChange={setTimeframe}>
            <SelectTrigger className="w-32 bg-white/5 border-white/10 text-white">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="24H">24 Hours</SelectItem>
              <SelectItem value="7D">7 Days</SelectItem>
              <SelectItem value="1M">1 Month</SelectItem>
              <SelectItem value="ALL">All Time</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </CardHeader>
      <CardContent className="p-6">
        {chartData.length === 0 ? (
          <div className="h-[400px] flex flex-col items-center justify-center space-y-4">
            <p className="text-white/60">No trading data available for the selected timeframe.</p>
            {timeframe !== 'ALL' && (
              <button 
                onClick={() => setTimeframe('ALL')}
                className="px-4 py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 transition-colors"
              >
                View All Data
              </button>
            )}
          </div>
        ) : (
          <div className="h-[400px] w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={chartData}
                margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="colorCumulative" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10B981" stopOpacity={0.1}/>
                    <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" />
                <XAxis
                  dataKey="timestamp"
                  stroke="rgba(255,255,255,0.5)"
                  tick={{ fill: 'rgba(255,255,255,0.5)' }}
                />
                <YAxis
                  stroke="rgba(255,255,255,0.5)"
                  tick={{ fill: 'rgba(255,255,255,0.5)' }}
                />
                <Tooltip content={<CustomTooltip />} />
                <Area
                  type="monotone"
                  dataKey="cumulative"
                  stroke="#10B981"
                  fillOpacity={1}
                  fill="url(#colorCumulative)"
                  isAnimationActive={false}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  )
}

export default TradingOverview

