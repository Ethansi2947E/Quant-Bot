import React, { useEffect, useState } from 'react'
import { AlertCircle } from 'lucide-react'
import { API_ENDPOINTS, fetchWithRetry, handleApiError } from './config'
import { TradingData, ProfitHistory } from './types'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import { format } from 'date-fns'

export function App() {
  const [tradingData, setTradingData] = useState<TradingData | null>(null)
  const [profitHistory, setProfitHistory] = useState<ProfitHistory | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [timeframe, setTimeframe] = useState<string>('7D')

  const fetchProfitHistory = async () => {
    try {
      console.log('Fetching profit history...')
      const data = await fetchWithRetry<ProfitHistory>(
        `${API_ENDPOINTS.PROFIT_HISTORY}?timeframe=${timeframe}`
      )
      
      console.log('Received profit history data:', data)
      
      // Detailed data validation
      if (!data) {
        throw new Error('Received null or undefined data')
      }
      
      if (!('data' in data)) {
        console.error('Missing data array in response:', data)
        throw new Error('Response missing data array. Expected format: { data: Array, cumulative_profit: number, timeframe: string }')
      }
      
      if (!Array.isArray(data.data)) {
        console.error('Data is not an array:', data.data)
        throw new Error('Data field is not an array. Expected array of profit entries.')
      }
      
      if (typeof data.cumulative_profit !== 'number') {
        console.error('Invalid cumulative_profit:', data.cumulative_profit)
        throw new Error('Invalid cumulative_profit. Expected number.')
      }
      
      // Validate each entry in the data array
      data.data.forEach((entry, index) => {
        if (!entry.timestamp || !entry.profit || typeof entry.cumulative !== 'number') {
          console.error(`Invalid entry at index ${index}:`, entry)
          throw new Error(`Invalid entry format at index ${index}. Expected: { timestamp: string, profit: number, cumulative: number }`)
        }
      })
      
      setProfitHistory(data)
    } catch (err) {
      console.error('Profit history error details:', {
        error: err,
        message: err instanceof Error ? err.message : 'Unknown error',
        stack: err instanceof Error ? err.stack : undefined,
      })
      const errorResponse = handleApiError(err)
      setError(errorResponse.error)
    }
  }

  const fetchTradingData = async () => {
    try {
      console.log('Fetching trading data...')
      setLoading(true)
      setError(null)
      const data = await fetchWithRetry<TradingData>(API_ENDPOINTS.TRADING_DATA)
      
      console.log('Received trading data:', data)
      
      // Detailed data validation
      if (!data) {
        throw new Error('Received null or undefined trading data')
      }
      
      const requiredFields = ['winRate', 'totalTrades', 'mt5_account']
      const missingFields = requiredFields.filter(field => !(field in data))
      
      if (missingFields.length > 0) {
        console.error('Missing required fields in trading data:', missingFields)
        throw new Error(`Missing required fields in trading data: ${missingFields.join(', ')}`)
      }
      
      if (typeof data.winRate !== 'number' || typeof data.totalTrades !== 'number') {
        console.error('Invalid field types:', {
          winRate: typeof data.winRate,
          totalTrades: typeof data.totalTrades
        })
        throw new Error('Invalid field types for winRate or totalTrades. Expected numbers.')
      }
      
      setTradingData(data)
      await fetchProfitHistory()
    } catch (err) {
      console.error('Trading data error details:', {
        error: err,
        message: err instanceof Error ? err.message : 'Unknown error',
        stack: err instanceof Error ? err.stack : undefined,
      })
      const errorResponse = handleApiError(err)
      setError(errorResponse.error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchTradingData()
    const interval = setInterval(fetchTradingData, 15000)
    return () => clearInterval(interval)
  }, [timeframe])

  const formatChartData = (data: ProfitHistory['data']) => {
    if (!Array.isArray(data)) return []
    
    return data.map(entry => ({
      ...entry,
      timestamp: new Date(entry.timestamp),
      formattedTime: format(new Date(entry.timestamp), 'HH:mm:ss'),
    }))
  }

  if (loading && !tradingData) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex h-screen items-center justify-center">
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 max-w-md">
          <div className="flex items-center">
            <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
            <p className="text-red-700">{error}</p>
          </div>
          <button 
            onClick={fetchTradingData}
            className="mt-3 px-4 py-2 bg-red-100 text-red-700 rounded-md hover:bg-red-200 transition-colors"
          >
            Retry Connection
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold">Profit History</h1>
          <select
            value={timeframe}
            onChange={(e) => setTimeframe(e.target.value)}
            className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 text-white"
          >
            <option value="24H">Last 24 Hours</option>
            <option value="7D">Last 7 Days</option>
            <option value="30D">Last 30 Days</option>
            <option value="ALL">All Time</option>
          </select>
        </div>
        
        {tradingData && profitHistory && (
          <div className="grid gap-6">
            {/* Stats Overview */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-gray-400 text-sm">Total Profit</h3>
                <p className={`text-2xl font-bold ${profitHistory.cumulative_profit >= 0 ? 'text-green-500' : 'text-red-500'}`}>
                  ${profitHistory.cumulative_profit.toFixed(2)}
                </p>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-gray-400 text-sm">Win Rate</h3>
                <p className="text-2xl font-bold text-blue-500">
                  {tradingData.winRate?.toFixed(1) || '0.0'}%
                </p>
              </div>
              <div className="bg-gray-800 rounded-lg p-4">
                <h3 className="text-gray-400 text-sm">Total Trades</h3>
                <p className="text-2xl font-bold text-white">
                  {tradingData.totalTrades || 0}
                </p>
              </div>
            </div>

            {/* Profit History Graph */}
            <div className="bg-gray-800 rounded-lg p-6">
              {profitHistory.data.length > 0 ? (
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={formatChartData(profitHistory.data)}
                      margin={{ top: 10, right: 30, left: 0, bottom: 0 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis
                        dataKey="formattedTime"
                        stroke="#9CA3AF"
                        tick={{ fill: '#9CA3AF' }}
                      />
                      <YAxis
                        stroke="#9CA3AF"
                        tick={{ fill: '#9CA3AF' }}
                        domain={['auto', 'auto']}
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1F2937',
                          border: 'none',
                          borderRadius: '0.375rem',
                          color: '#F9FAFB'
                        }}
                        labelStyle={{ color: '#9CA3AF' }}
                      />
                      <ReferenceLine y={0} stroke="#4B5563" />
                      <Line
                        type="monotone"
                        dataKey="cumulative"
                        stroke="#10B981"
                        dot={false}
                        strokeWidth={2}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <div className="flex items-center justify-center h-[400px] text-gray-400">
                  No profit history data available
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
} 