"use client"

import { useState, useEffect } from "react"
import TradingOverview from "@/components/TradingOverview"
import ActiveTrades from "@/components/ActiveTrades"
import { API_ENDPOINTS } from "@/app/config"
import {
  LayoutDashboard,
  BarChart2,
  List,
  TrendingUp,
  Settings,
  Bell,
  ChevronUp,
  Wallet,
  Activity,
  Percent,
  AlertTriangle,
} from "lucide-react"
import { Card } from "@/components/ui/card"
import ConfigurationSettings from "@/components/ConfigurationSettings"
import { PerformanceAnalytics } from "@/components/PerformanceAnalytics"
import { MarketStatus } from "@/components/MarketStatus"

// Mock data
const mockTradingStatus = {
  total_profit: 15000,
  daily_profit: 1200,
  win_rate: 68,
  total_trades: 150,
  trading_status: "Enabled",
  active_trades: [
    { symbol: "EURUSD", type: "BUY", entry: 1.185, sl: 1.183, tp: 1.188, profit: 150 },
    { symbol: "GBPJPY", type: "SELL", entry: 150.5, sl: 150.8, tp: 150.0, profit: -75 },
    { symbol: "USDCAD", type: "BUY", entry: 1.256, sl: 1.254, tp: 1.259, profit: 100 },
  ],
  profit_history: Array.from({ length: 24 }, (_, i) => ({
    timestamp: new Date(Date.now() - (23 - i) * 3600000).toISOString(),
    profit: 14000 + Math.random() * 2000,
  })),
  market_analysis: {
    market_bias: "Bullish",
    structure_type: "Uptrend",
    key_levels: [
      { price: 1.185, type: "support" },
      { price: 1.19, type: "resistance" },
    ],
  },
}

export default function Dashboard() {
  const [tradingStatus, setTradingStatus] = useState<any>(null)
  const [activePage, setActivePage] = useState("dashboard")
  const [error, setError] = useState<string | null>(null)
  const [isTogglingTrading, setIsTogglingTrading] = useState(false)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(API_ENDPOINTS.TRADING_DATA)
        if (!response.ok) {
          throw new Error('Failed to fetch trading data')
        }
        const data = await response.json()
        setTradingStatus(data)
        setError(null)
      } catch (err) {
        console.error("Error fetching data:", err)
        setError("Failed to fetch trading data. Please try again later.")
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 5000)  // Refresh every 5 seconds
    return () => clearInterval(interval)
  }, [])

  const renderContent = () => {
    if (error) {
      return (
        <Card className="p-6 bg-red-500/10 backdrop-blur-xl border-red-500/20">
          <div className="flex items-center gap-4 text-red-500">
            <AlertTriangle className="h-6 w-6" />
            <p>{error}</p>
          </div>
        </Card>
      )
    }

    if (!tradingStatus) {
      return (
        <Card className="p-6 bg-white/5 backdrop-blur-xl border-white/10">
          <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-white"></div>
          </div>
        </Card>
      )
    }

    switch (activePage) {
      case "dashboard":
        return (
          <div className="space-y-8">
            {/* Market Status */}
            <MarketStatus />
            
            {/* Stats Overview */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <Card className="p-6 bg-white/5 backdrop-blur-xl border-white/10">
                <div className="flex items-center gap-4">
                  <div className="h-14 w-14 rounded-2xl bg-blue-500/10 flex items-center justify-center">
                    <Wallet className="h-8 w-8 text-blue-500" />
                  </div>
                  <div>
                    <p className="text-sm text-white/80">MT5 Balance</p>
                    <h3 className="text-3xl font-bold text-white mt-1">
                      ${tradingStatus?.mt5_account?.balance?.toLocaleString() || '0.00'}
                    </h3>
                    <p className="text-sm text-green-400 flex items-center gap-1 mt-2">
                      <ChevronUp className="h-5 w-5" />
                      Equity: ${tradingStatus?.mt5_account?.equity?.toLocaleString() || '0.00'}
                    </p>
                  </div>
                </div>
              </Card>

              <Card className="p-6 bg-white/5 backdrop-blur-xl border-white/10">
                <div className="flex items-center gap-4">
                  <div className="h-14 w-14 rounded-2xl bg-green-500/10 flex items-center justify-center">
                    <TrendingUp className="h-8 w-8 text-green-500" />
                  </div>
                  <div>
                    <p className="text-sm text-white/80">Current Profit</p>
                    <h3 className="text-3xl font-bold text-white mt-1">
                      ${tradingStatus?.mt5_account?.profit?.toLocaleString() || '0.00'}
                    </h3>
                    <p className="text-sm text-white/80 mt-2">
                      Free Margin: ${tradingStatus?.mt5_account?.margin_free?.toLocaleString() || '0.00'}
                    </p>
                  </div>
                </div>
              </Card>

              <Card className="p-6 bg-white/5 backdrop-blur-xl border-white/10">
                <div className="flex items-center gap-4">
                  <div className="h-14 w-14 rounded-2xl bg-purple-500/10 flex items-center justify-center">
                    <Percent className="h-8 w-8 text-purple-500" />
                  </div>
                  <div>
                    <p className="text-sm text-white/80">Win Rate</p>
                    <h3 className="text-3xl font-bold text-white mt-1">{tradingStatus?.win_rate || 0}%</h3>
                    <p className="text-sm text-white/80 mt-2">
                      Margin Level: {tradingStatus?.mt5_account?.margin_level?.toFixed(2) || '0.00'}%
                    </p>
                  </div>
                </div>
              </Card>

              <Card className="p-6 bg-white/5 backdrop-blur-xl border-white/10">
                <div className="flex items-center gap-4">
                  <div className="h-14 w-14 rounded-2xl bg-green-500/10 flex items-center justify-center">
                    <Activity className="h-8 w-8 text-green-500" />
                  </div>
                  <div>
                    <p className="text-sm text-white/80">Trading Status</p>
                    <h3 className="text-3xl font-bold text-white flex items-center gap-2 mt-1">
                      {tradingStatus?.trading_status || 'Disabled'}
                      <span className="text-sm px-3 py-1 rounded-full bg-green-500/20 text-green-400">
                        {tradingStatus?.active_trades?.length || 0} Active
                      </span>
                    </h3>
                  </div>
                </div>
              </Card>
            </div>

            {/* Trading Overview & Active Trades */}
            <div className="grid grid-cols-1 gap-8">
              {/* Profit History Chart */}
              <TradingOverview />

              {/* Active Trades */}
              <Card className="p-6 bg-white/5 backdrop-blur-xl border-white/10">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-2xl font-bold">Active Trades</h2>
                  <span className="px-3 py-1 rounded-full text-base font-medium bg-blue-500/20 text-blue-500">
                    {tradingStatus?.active_trades?.length || 0} Positions
                  </span>
                </div>
                <ActiveTrades data={tradingStatus?.active_trades || []} />
              </Card>
            </div>
          </div>
        )
      case "performance":
        return (
          <div className="space-y-4">
            <PerformanceAnalytics />
          </div>
        )
      case "trades":
        return <ActiveTrades data={tradingStatus.active_trades} />
      case "settings":
        return (
          <div className="space-y-8">
            <ConfigurationSettings
              onSave={async (config) => {
                try {
                  const response = await fetch(`${API_ENDPOINTS.TRADING_DATA}/config`, {
                    method: 'POST',
                    headers: {
                      'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(config),
                  })
                  
                  if (!response.ok) {
                    throw new Error('Failed to update configuration')
                  }
                  
                  // Refetch trading data to reflect changes
                  const updatedData = await fetch(API_ENDPOINTS.TRADING_DATA).then(res => res.json())
                  setTradingStatus(updatedData)
                } catch (err) {
                  console.error("Error saving configuration:", err)
                  setError("Failed to save configuration. Please try again later.")
                }
              }}
            />
          </div>
        )
      default:
        return null
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-900 to-gray-800">
      <div className="flex">
        {/* Sidebar */}
        <aside className="fixed inset-y-0 left-0 w-64 bg-white/5 backdrop-blur-xl border-r border-white/10">
          <div className="flex flex-col h-full">
            <div className="p-6">
              <div className="flex items-center gap-3">
                <div className="h-12 w-12 rounded-2xl bg-gradient-to-br from-blue-500 to-blue-600 flex items-center justify-center">
                  <TrendingUp className="h-8 w-8 text-white" />
                </div>
                <h1 className="text-2xl font-bold bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
                  TradeMaster AI
                </h1>
              </div>
            </div>

            <nav className="flex-1 px-4 py-8">
              <div className="space-y-2">
                {[
                  { icon: LayoutDashboard, label: "Dashboard", id: "dashboard" },
                  { icon: BarChart2, label: "Performance", id: "performance" },
                  { icon: List, label: "Active Trades", id: "trades" },
                  { icon: Settings, label: "Settings", id: "settings" },
                ].map((item) => (
                  <button
                    key={item.id}
                    onClick={() => setActivePage(item.id)}
                    className={`w-full flex items-center gap-3 px-4 py-3 text-base font-medium rounded-xl transition-colors
                      ${
                        activePage === item.id
                          ? "bg-white/10 text-white"
                          : "text-white/60 hover:bg-white/5 hover:text-white"
                      }`}
                  >
                    <item.icon className="h-6 w-6" />
                    {item.label}
                  </button>
                ))}
              </div>
            </nav>

            <div className="p-4">
              <Card className="bg-white/5 border-white/10 p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-base font-medium text-white/80">Bot Status</span>
                  <button
                    onClick={async () => {
                      if (isTogglingTrading) return; // Prevent double-clicks
                      setIsTogglingTrading(true);
                      try {
                        // First try the new toggle endpoint
                        try {
                          const response = await fetch(API_ENDPOINTS.TOGGLE_TRADING, {
                            method: 'POST',
                            headers: {
                              'Content-Type': 'application/json'
                            }
                          });
                          
                          const result = await response.json();
                          
                          if (result.status === 'success') {
                            setError(null);
                            if (result.data) {
                              setTradingStatus(result.data);
                            } else {
                              // Fallback: fetch fresh data
                              const newDataResponse = await fetch(API_ENDPOINTS.TRADING_DATA);
                              if (!newDataResponse.ok) {
                                throw new Error('Failed to fetch updated trading data');
                              }
                              const newData = await newDataResponse.json();
                              setTradingStatus(newData);
                            }
                            return;
                          } else {
                            throw new Error(result.message || 'Failed to toggle trading');
                          }
                        } catch (toggleError) {
                          console.warn('Toggle endpoint failed, trying fallback:', toggleError);
                          
                          // Fallback to individual enable/disable endpoints
                          const currentState = tradingStatus?.trading_status === 'Enabled';
                          const endpoint = currentState ? API_ENDPOINTS.DISABLE_TRADING : API_ENDPOINTS.ENABLE_TRADING;
                          
                          const response = await fetch(endpoint, {
                            method: 'POST',
                            headers: {
                              'Content-Type': 'application/json'
                            }
                          });
                          
                          if (!response.ok) {
                            throw new Error(`Failed to ${currentState ? 'disable' : 'enable'} trading`);
                          }
                          
                          const result = await response.json();
                          if (result.status === 'success') {
                            setError(null);
                            // Fetch fresh trading data
                            const newDataResponse = await fetch(API_ENDPOINTS.TRADING_DATA);
                            if (!newDataResponse.ok) {
                              throw new Error('Failed to fetch updated trading data');
                            }
                            const newData = await newDataResponse.json();
                            setTradingStatus(newData);
                          } else {
                            throw new Error(result.message || `Failed to ${currentState ? 'disable' : 'enable'} trading`);
                          }
                        }
                      } catch (err) {
                        console.error('Error toggling trading:', err);
                        setError(err instanceof Error ? err.message : "Failed to toggle trading status. Please check if the server is running.");
                        
                        // Try to refresh trading status
                        try {
                          const statusResponse = await fetch(API_ENDPOINTS.TRADING_DATA);
                          if (statusResponse.ok) {
                            const currentStatus = await statusResponse.json();
                            setTradingStatus(currentStatus);
                          }
                        } catch (refreshErr) {
                          console.error('Error refreshing trading status:', refreshErr);
                        }
                      } finally {
                        setIsTogglingTrading(false);
                      }
                    }}
                    disabled={isTogglingTrading}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 ${
                      tradingStatus?.trading_status === 'Enabled' ? 'bg-green-500' : 'bg-gray-500'
                    } ${isTogglingTrading ? 'opacity-50 cursor-not-allowed' : ''}`}
                  >
                    <span className={`${
                      tradingStatus?.trading_status === 'Enabled' ? 'translate-x-6' : 'translate-x-1'
                    } inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                      isTogglingTrading ? 'animate-pulse' : ''
                    }`} />
                  </button>
                </div>
                <div className="flex items-center gap-2">
                  <div className={`h-3 w-3 rounded-full ${
                    tradingStatus?.trading_status === 'Enabled' ? 'bg-green-500 animate-pulse' : 'bg-gray-500'
                  }`}></div>
                  <span className={`text-sm font-medium ${
                    tradingStatus?.trading_status === 'Enabled' ? 'text-green-500' : 'text-gray-500'
                  }`}>
                    {tradingStatus?.trading_status === 'Enabled' ? 'Running' : 'Stopped'}
                  </span>
                </div>
              </Card>
            </div>
          </div>
        </aside>

        {/* Main content */}
        <main className="flex-1 ml-64">
          <div className="max-w-7xl mx-auto p-8">
            <header className="flex justify-between items-center mb-8">
              <div>
                <h2 className="text-4xl font-bold bg-gradient-to-r from-white to-white/60 bg-clip-text text-transparent">
                  {activePage.charAt(0).toUpperCase() + activePage.slice(1)}
                </h2>
                <p className="text-xl text-white/80 mt-2">
                  {activePage === "dashboard"
                    ? "Real-time trading performance and analytics"
                    : `View and manage your ${activePage}`}
                </p>
              </div>
              <div className="flex items-center gap-4">
                <button className="p-2 rounded-xl bg-white/5 hover:bg-white/10 transition-colors">
                  <Bell className="h-6 w-6 text-white/80" />
                </button>
                <div className="flex items-center gap-2 px-3 py-2 rounded-xl bg-green-500/10">
                  <div className="h-3 w-3 rounded-full bg-green-500 animate-pulse"></div>
                  <span className="text-base font-medium text-green-500">Live Data</span>
                </div>
              </div>
            </header>

            {renderContent()}
          </div>
        </main>
      </div>
    </div>
  )
}
